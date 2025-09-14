#!/usr/bin/env python3
# DDP-aware Transformer training

import os, sys, argparse, logging, random, numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

from data_utils import DataProcessor, TripDataset
from model_utils import TrajectoryModel

# --------- DDP helpers ---------
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
# --------------------------------

def setup_logger(log_file='train.log'):
    """Rank-aware logger: only rank 0 emits logs to console/file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        return logger
    if is_primary():
        fh = logging.FileHandler(log_file); fh.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
    else:
        logger.addHandler(logging.NullHandler()); logger.setLevel(logging.ERROR)
    return logger

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def maybe_get_core_module(model: TrajectoryModel):
    if hasattr(model, "net") and isinstance(getattr(model, "net"), torch.nn.Module):
        return "net", model.net
    if hasattr(model, "model") and isinstance(getattr(model, "model"), torch.nn.Module):
        return "model", model.model
    return None, None

def main():
    ddp_setup()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train a Trajectory Transformer model.")

    # --- Data ---
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--feature_columns', nargs='+', default=['speed'])
    parser.add_argument('--target_column', type=str, default='label')
    parser.add_argument('--traj_id_column', type=str, default='traj_id')
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--random_state', type=int, default=316)
    parser.add_argument('--chunksize', type=int, default=10**6)
    parser.add_argument('--window_size', type=int, default=100)   # <-- tuned
    parser.add_argument('--stride', type=int, default=25)         # <-- tuned

    # --- Training ---
    parser.add_argument('--batch_size', type=int, default=128)    # <-- tuned (per-GPU)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--kv_heads', type=int, default=6, help='GQA KV heads')  # <-- tuned

    # --- Model hparams ---
    parser.add_argument('--d_model', type=int, default=384)       # <-- tuned
    parser.add_argument('--nhead', type=int, default=12)          # <-- tuned
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)     # <-- tuned

    # --- Optimizer ---
    parser.add_argument('--learning_rate', type=float, default=5.64070780267346e-05)  # <-- tuned
    parser.add_argument('--weight_decay', type=float, default=1e-4)                    # <-- tuned
    parser.add_argument('--gradient_clip', type=float, default=1.0)

    # --- Saving paths ---
    parser.add_argument('--save_model_path', type=str, default='best_model.pth')
    parser.add_argument('--save_label_encoder_path', type=str, default='label_encoder.joblib')
    parser.add_argument('--save_scaler_path', type=str, default='scaler.joblib')

    # --- AMP ---
    parser.add_argument('--use_amp', action='store_true', default=False)  # tuned run used amp=0

    # --- Sweep bookkeeping ---
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--out_dir',  type=str, default='sweeps')

    args = parser.parse_args()

    # Per-run folder + logger (after parse)
    run_name = args.run_name or "run"
    run_dir  = os.path.join(args.out_dir, run_name)
    if is_primary():
        os.makedirs(run_dir, exist_ok=True)
        args.save_model_path         = os.path.join(run_dir, 'best_model.pth')
        args.save_label_encoder_path = os.path.join(run_dir, 'label_encoder.joblib')
        args.save_scaler_path        = os.path.join(run_dir, 'scaler.joblib')
    log_path = os.path.join(run_dir, 'train.log') if is_primary() else 'train.log'
    logger = setup_logger(log_path)

    # Seeds & DDP info
    if is_primary():
        logger.info(f"Setting random seed to {args.random_state}")
        logger.info(f"DDP world_size={world_size()} local_rank={local_rank}")
    set_seed(args.random_state)

    # 1) Data
    if is_primary(): logger.info("Initializing DataProcessor...")
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
    if is_primary():
        processor.save_preprocessors(args.save_scaler_path, args.save_label_encoder_path)
        logger.info(f"Scaler saved to {args.save_scaler_path}")
        logger.info(f"Label encoder saved to {args.save_label_encoder_path}")

    train_dataset = TripDataset(processor.train_sequences, processor.train_labels, processor.train_masks)
    val_dataset   = TripDataset(processor.val_sequences,   processor.val_labels,   processor.val_masks)
    test_dataset  = TripDataset(processor.test_sequences,  processor.test_labels,  processor.test_masks)

    if world_size() > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True,  drop_last=False)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False, drop_last=False)
        test_sampler  = DistributedSampler(test_dataset,  shuffle=False, drop_last=False)
        shuffle_train = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle_train = True

    g = torch.Generator(); g.manual_seed(args.random_state + 10_000 * int(os.environ.get("RANK", "0")))
    def worker_init_fn(worker_id):
        base = args.random_state + 1000 * int(os.environ.get("RANK", "0"))
        ss = base + worker_id
        np.random.seed(ss); random.seed(ss); torch.manual_seed(ss)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle_train, sampler=train_sampler,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn, generator=g, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=max(1, args.batch_size * 2), shuffle=False, sampler=test_sampler,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn, pin_memory=True)

    if is_primary(): logger.info("Datasets and DataLoaders ready (DDP-aware).")

    # 2) Model
    if is_primary(): logger.info("Initializing TrajectoryModel...")
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
        weight_decay=args.weight_decay,
        kv_heads=args.kv_heads
    )

    attr_name, core = maybe_get_core_module(model)
    core = core if core is not None else None
    if core is None:
        if is_primary():
            logger.warning("Could not locate underlying nn.Module inside TrajectoryModel; DDP wrapping skipped.")
    else:
        core.to(device)
        if world_size() > 1:
            wrapped = DDP(core, device_ids=[local_rank], output_device=local_rank)
            setattr(model, attr_name, wrapped)

    if is_primary(): logger.info("Model initialization completed.")

    # 3) Train
    best_val = float("inf"); epochs_no_improve = 0
    if is_primary(): logger.info("Starting training loop...")
    for epoch in range(args.num_epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"): train_sampler.set_epoch(epoch)
        if val_sampler   is not None and hasattr(val_sampler,   "set_epoch"): val_sampler.set_epoch(epoch)

        tr_loss, tr_acc = model.train_one_epoch(train_loader, gradient_clip=args.gradient_clip)
        va_loss, va_acc = model.evaluate(val_loader)

        tr_loss = reduce_mean_scalar(tr_loss, device)
        tr_acc  = reduce_mean_scalar(tr_acc,  device)
        va_loss = reduce_mean_scalar(va_loss, device)
        va_acc  = reduce_mean_scalar(va_acc,  device)

        if is_primary():
            logger.info(f"[Epoch {epoch+1}/{args.num_epochs}] "
                        f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, "
                        f"Val Loss: {va_loss:.4f}, Val Acc: {va_acc:.4f}")
            if va_loss < best_val:
                best_val = va_loss; epochs_no_improve = 0
                to_save = None
                if attr_name and isinstance(getattr(model, attr_name), DDP):
                    to_save = getattr(model, attr_name).module.state_dict()
                elif attr_name:
                    to_save = getattr(model, attr_name).state_dict()
                if to_save is not None: torch.save(to_save, args.save_model_path)
                elif hasattr(model, "save_model"): model.save_model(args.save_model_path)
                logger.info("  -> Best model saved")
            else:
                epochs_no_improve += 1
                logger.info(f"  -> No improvement ({epochs_no_improve}/{args.patience})")
                if epochs_no_improve >= args.patience:
                    logger.info("Early stopping triggered."); break

    # 4) Test (rank 0 only)
    if is_primary():
        logger.info("\nLoading best model and evaluating on test set...")
        if attr_name:
            core_mod = getattr(model, attr_name)
            target = core_mod.module if isinstance(core_mod, DDP) else core_mod
            state = torch.load(args.save_model_path, map_location=device)
            target.load_state_dict(state)
        elif hasattr(model, "load_model"):
            model.load_model(args.save_model_path)

        te_loss, te_acc = model.evaluate(test_loader)
        te_loss = reduce_mean_scalar(te_loss, device)
        te_acc  = reduce_mean_scalar(te_acc,  device)
        logger.info(f"Test Loss: {te_loss:.4f}, Test Accuracy: {te_acc:.4f}")

        all_labels, all_preds = model.predict(test_loader)
        report = classification_report(all_labels, all_preds, target_names=processor.label_encoder.classes_)
        logger.info("Classification Report:\n" + report)

        cm = confusion_matrix(all_labels, all_preds)
        logger.info("Saving confusion matrix as confusion_matrix.png...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=processor.label_encoder.classes_,
                    yticklabels=processor.label_encoder.classes_)
        plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.title('Confusion Matrix')
        plt.tight_layout(); plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'), dpi=200)

    ddp_cleanup()

if __name__ == "__main__":
    main()