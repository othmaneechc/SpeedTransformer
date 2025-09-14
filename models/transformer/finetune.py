#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import random
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from data_utils import DataProcessor, TripDataset
from model_utils import TrajectoryModel


# -------------------- DDP helpers --------------------
def ddp_setup():
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_primary() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

@torch.no_grad()
def reduce_mean_scalar(x: float, device: torch.device) -> float:
    t = torch.tensor([x], device=device, dtype=torch.float64)
    if world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size()
    return float(t.item())

def bcast_bool(flag: bool, device: torch.device) -> bool:
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int)
    if world_size() > 1:
        dist.broadcast(t, src=0)
    return bool(int(t.item()))
# -----------------------------------------------------


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file='finetune.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        return logger
    fh = logging.FileHandler(log_file); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def main():
    ddp_setup()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logger = setup_logger('finetune.log')

    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained Trajectory Transformer (DDP-aware).")
    # Data & splits
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--feature_columns', nargs='+', default=['speed'])
    parser.add_argument('--target_column', type=str, default='label')
    parser.add_argument('--traj_id_column', type=str, default='traj_id')
    parser.add_argument('--test_size', type=float, default=0.7)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--chunksize', type=int, default=10**6)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--stride', type=int, default=50)

    # Loaders
    parser.add_argument('--batch_size', type=int, default=1024)  # per-GPU
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)

    # Model Architecture
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--kv_heads', type=int, default=4, help='Number of key/value heads')

    # Fine-tuning Control
    parser.add_argument('--freeze_embeddings', action='store_true', 
                      help='Freeze the embedding layer')
    parser.add_argument('--freeze_layers', type=str, default='',
                      help='Comma-separated list of transformer layer indices to freeze (e.g., "0,1,2")')
    parser.add_argument('--freeze_attention', action='store_true',
                      help='Freeze attention mechanisms in all layers')
    parser.add_argument('--freeze_feedforward', action='store_true',
                      help='Freeze feedforward networks in all layers')
    parser.add_argument('--reinit_layers', type=str, default='',
                      help='Comma-separated list of transformer layer indices to reinitialize')
    
    # Optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_clip', type=float, default=0.5)
    parser.add_argument('--warmup_steps', type=int, default=0,
                      help='Number of warmup steps for learning rate')
    parser.add_argument('--layer_lr_decay', type=float, default=1.0,
                      help='Per-layer learning rate decay factor (1.0 = disabled)')

    # Checkpoints / preprocessors
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--save_model_path', type=str, default='finetuned_model.pth')
    parser.add_argument('--label_encoder_path', type=str, default='label_encoder.joblib')

    # Optional AMP to match your TrajectoryModel
    parser.add_argument('--use_amp', action='store_true')

    args = parser.parse_args()
    set_global_seed(args.random_state)

    # -----------------------------
    # 1) Data processing
    # -----------------------------
    if is_primary():
        logger.info("Initializing DataProcessor for fine-tuning...")
    processor = DataProcessor(
        data_path=args.data_path,
        feature_columns=args.feature_columns,
        target_column=args.target_column,
        traj_id_column=args.traj_id_column,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        chunksize=args.chunksize,
        window_size=args.window_size,
        stride=args.stride
    )
    processor.load_and_process_data()

    # Use the saved label encoder (keeps class indices consistent with pretraining)
    processor.label_encoder = joblib.load(args.label_encoder_path)

    # Recreate sequences after replacing encoder
    processor.train_sequences.clear(); processor.train_labels.clear(); processor.train_masks.clear()
    processor.val_sequences.clear();   processor.val_labels.clear();   processor.val_masks.clear()
    processor.test_sequences.clear();  processor.test_labels.clear();  processor.test_masks.clear()
    processor.create_sequences()

    # Datasets
    train_ds = TripDataset(processor.train_sequences, processor.train_labels, processor.train_masks)
    val_ds   = TripDataset(processor.val_sequences,   processor.val_labels,   processor.val_masks)
    test_ds  = TripDataset(processor.test_sequences,  processor.test_labels,  processor.test_masks)

    # Distributed samplers
    if world_size() > 1:
        train_samp = DistributedSampler(train_ds, shuffle=True,  drop_last=False)
        val_samp   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
        test_samp  = DistributedSampler(test_ds,  shuffle=False, drop_last=False)
    else:
        train_samp = val_samp = test_samp = None

    # Deterministic worker init
    def worker_init_fn(worker_id):
        base = args.random_state + 1000 * int(os.environ.get("RANK", "0"))
        seed = base + worker_id
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    # DataLoaders (batch_size is per-GPU)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_samp is None),
        sampler=train_samp, num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_samp, num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        sampler=test_samp, num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # -----------------------------
    # 2) Load model & wrap with DDP
    # -----------------------------
    model = TrajectoryModel(
        feature_columns=args.feature_columns,
        label_encoder=processor.label_encoder,
        use_amp=args.use_amp
    )
    model.prepare_model(
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if is_primary():
        logger.info(f"Loading pretrained model from: {args.pretrained_model_path}")
    model.load_model(args.pretrained_model_path)

    # Apply fine-tuning controls
    def get_layer_indices(layer_str):
        return [int(i) for i in layer_str.split(',')] if layer_str else []

    # 1. Process freezing options
    freeze_layers = get_layer_indices(args.freeze_layers)
    
    # Get transformer layers for easier access
    transformer_layers = model.model.encoder.layers    # Embeddings
    if args.freeze_embeddings:
        for p in model.model.embedding.parameters():
            p.requires_grad = False
        if is_primary():
            logger.info("Frozen: Embedding layer")
    
    # Layer-specific freezing
    for idx, layer in enumerate(transformer_layers):
        if idx in freeze_layers:
            for p in layer.parameters():
                p.requires_grad = False
            if is_primary():
                logger.info(f"Frozen: Entire layer {idx}")
        else:
            # Selective mechanism freezing
            if args.freeze_attention:
                for p in layer.attn.parameters():
                    p.requires_grad = False
                if is_primary() and idx == 0:
                    logger.info("Frozen: Attention mechanisms in all unfrozen layers")
            
            if args.freeze_feedforward:
                for p in layer.ffn.up.parameters():
                    p.requires_grad = False
                for p in layer.ffn.down.parameters():
                    p.requires_grad = False
                if is_primary() and idx == 0:
                    logger.info("Frozen: Feedforward networks in all unfrozen layers")
    
    # 2. Layer reinitialization
    reinit_layers = get_layer_indices(args.reinit_layers)
    for idx in reinit_layers:
        if idx < len(transformer_layers):
            layer = transformer_layers[idx]
            
            # Attention weights
            if hasattr(layer.attn, 'q_proj'):
                torch.nn.init.xavier_uniform_(layer.attn.q_proj.weight)
                torch.nn.init.xavier_uniform_(layer.attn.k_proj.weight)
                torch.nn.init.xavier_uniform_(layer.attn.v_proj.weight)
                torch.nn.init.xavier_uniform_(layer.attn.out.weight)
                if layer.attn.q_proj.bias is not None:
                    torch.nn.init.zeros_(layer.attn.q_proj.bias)
                    torch.nn.init.zeros_(layer.attn.k_proj.bias)
                    torch.nn.init.zeros_(layer.attn.v_proj.bias)
                    torch.nn.init.zeros_(layer.attn.out.bias)
            
            # FFN weights
            if hasattr(layer.ffn, 'up'):
                torch.nn.init.xavier_uniform_(layer.ffn.up.weight)
                torch.nn.init.xavier_uniform_(layer.ffn.down.weight)
                if layer.ffn.up.bias is not None:
                    torch.nn.init.zeros_(layer.ffn.up.bias)
                if layer.ffn.down.bias is not None:
                    torch.nn.init.zeros_(layer.ffn.down.bias)
            
            # Layer norms (use standard initialization)
            if hasattr(layer, 'norm1'):
                torch.nn.init.ones_(layer.norm1.weight)
                torch.nn.init.zeros_(layer.norm1.bias)
            if hasattr(layer, 'norm2'):
                torch.nn.init.ones_(layer.norm2.weight)
                torch.nn.init.zeros_(layer.norm2.bias)
                
            if is_primary():
                logger.info(f"Reinitialized: Layer {idx}")
    
    # 3. Setup optimizer with layer-wise learning rates
    param_groups = []
    
    # Embedding group (if not frozen)
    if not args.freeze_embeddings:
        param_groups.append({
            'params': model.model.embedding.parameters(),
            'lr': args.learning_rate
        })
    
    # Layer groups with decay
    num_layers = len(transformer_layers)
    for idx, layer in enumerate(transformer_layers):
        if idx not in freeze_layers:
            # Apply layer-wise learning rate decay
            layer_lr = args.learning_rate * (args.layer_lr_decay ** (num_layers - idx - 1))
            param_groups.append({
                'params': layer.parameters(),
                'lr': layer_lr
            })
            if is_primary() and args.layer_lr_decay != 1.0:
                logger.info(f"Layer {idx} learning rate: {layer_lr:.2e}")
    
    # Create optimizer with param groups
    model.optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # 4. Add warmup scheduler if requested
    if args.warmup_steps > 0:
        def lr_lambda(step):
            if step < args.warmup_steps:
                return float(step) / float(max(1, args.warmup_steps))
            return 1.0
        
        model.scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda)
        if is_primary():
            logger.info(f"Added warmup scheduler with {args.warmup_steps} steps")

    # Wrap core module with DDP
    core = model.model.to(device)
    if world_size() > 1:
        model.model = DDP(core, device_ids=[local_rank], output_device=local_rank)

    # -----------------------------
    # 3) Fine-tuning loop (global metrics + early stop)
    # -----------------------------
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(args.num_epochs):
        if train_samp is not None:
            train_samp.set_epoch(epoch)

        if is_primary():
            logger.info(f"[Epoch {epoch+1}/{args.num_epochs}]")

        train_loss, train_acc = model.train_one_epoch(train_loader, gradient_clip=args.gradient_clip)
        
        # Step LR scheduler if using warmup
        if args.warmup_steps > 0:
            model.scheduler.step()
            
        val_loss, val_acc = model.evaluate(val_loader)

        # Average across GPUs
        train_loss = reduce_mean_scalar(train_loss, device)
        train_acc  = reduce_mean_scalar(train_acc,  device)
        val_loss   = reduce_mean_scalar(val_loss,   device)
        val_acc    = reduce_mean_scalar(val_acc,    device)
        
        # Log learning rates if using layer-wise decay
        if is_primary() and args.layer_lr_decay != 1.0:
            for idx, group in enumerate(model.optimizer.param_groups):
                logger.info(f"Group {idx} current lr: {group['lr']:.2e}")

        if is_primary():
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        # Early stopping on rank 0
        improved = False
        if is_primary():
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                # Save only on rank 0
                model.save_model(args.save_model_path)
                logger.info("  -> Best fine-tuned model saved.")
                improved = True
            else:
                epochs_no_improve += 1
                logger.info(f"  -> No improvement ({epochs_no_improve}/{args.patience})")

        # Broadcast early-stop decision
        stop = False
        if is_primary():
            stop = epochs_no_improve >= args.patience
        stop = bcast_bool(stop, device)
        if stop:
            if is_primary():
                logger.info("Early stopping triggered.")
            break

    # -----------------------------
    # 4) Final evaluation (rank 0 report)
    # -----------------------------
    if is_primary():
        logger.info(f"\nLoading best fine-tuned model from {args.save_model_path} for final evaluation...")
    model.load_model(args.save_model_path)  # safe on all ranks; only rank0 saved

    # Evaluate (per-rank), then reduce
    test_loss, test_acc = model.evaluate(test_loader)
    test_loss = reduce_mean_scalar(test_loss, device)
    test_acc  = reduce_mean_scalar(test_acc,  device)

    # Gather predictions from all ranks for detailed report
    local_labels, local_preds = model.predict(test_loader)
    if world_size() > 1:
        gathered_labels = [None for _ in range(world_size())]
        gathered_preds  = [None for _ in range(world_size())]
        dist.all_gather_object(gathered_labels, local_labels)
        dist.all_gather_object(gathered_preds,  local_preds)
        all_labels, all_preds = [], []
        for la, pr in zip(gathered_labels, gathered_preds):
            all_labels.extend(la); all_preds.extend(pr)
    else:
        all_labels, all_preds = local_labels, local_preds

    if is_primary():
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        report = classification_report(all_labels, all_preds, target_names=processor.label_encoder.classes_)
        logger.info("Classification Report:\n" + report)

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=processor.label_encoder.classes_,
                    yticklabels=processor.label_encoder.classes_)
        plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.title('Confusion Matrix (Fine-Tuned)')
        plt.tight_layout()
        plt.savefig('confusion_matrix_finetune.png', dpi=200)
        logger.info("Saved confusion_matrix_finetune.png")

    ddp_cleanup()


if __name__ == "__main__":
    main()