#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from data_utils import DataProcessor, TripDataset
from model_utils import TrajectoryModel


# ---------- DDP helpers ----------
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
# ---------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ddp_setup()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Test a Trajectory Transformer model (DDP-aware).")
    # Data / splits
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--feature_columns', nargs='+', default=['speed'])
    parser.add_argument('--target_column', type=str, default='label')
    parser.add_argument('--traj_id_column', type=str, default='traj_id')
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--chunksize', type=int, default=10**6)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--stride', type=int, default=50)

    # Loader
    parser.add_argument('--batch_size', type=int, default=512)      # per-GPU
    parser.add_argument('--num_workers', type=int, default=4)

    # Model hparams (must match training)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Weights / preprocessors
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--label_encoder_path', type=str, default='label_encoder.joblib')

    # Optional AMP (matches your TrajectoryModel flag)
    parser.add_argument('--use_amp', action='store_true')

    args = parser.parse_args()

    # Seeds
    set_seed(args.random_state)

    # -----------------------------
    # 1) Data processing
    # -----------------------------
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

    # Use the saved label encoder to guarantee class ordering
    processor.label_encoder = joblib.load(args.label_encoder_path)

    # Rebuild sequences with the loaded encoder
    processor.train_sequences.clear(); processor.train_labels.clear(); processor.train_masks.clear()
    processor.val_sequences.clear();   processor.val_labels.clear();   processor.val_masks.clear()
    processor.test_sequences.clear();  processor.test_labels.clear();  processor.test_masks.clear()
    processor.create_sequences()

    test_dataset = TripDataset(processor.test_sequences, processor.test_labels, processor.test_masks)

    # DDP sampler
    if world_size() > 1:
        test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)
    else:
        test_sampler = None

    # Rank-differentiated worker seeding
    def worker_init_fn(worker_id):
        base = args.random_state + 1000 * int(os.environ.get("RANK", "0"))
        ss = base + worker_id
        np.random.seed(ss); random.seed(ss); torch.manual_seed(ss)

    test_loader = DataLoader(
        test_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    # -----------------------------
    # 2) Initialize & Load Model
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
        dropout=args.dropout
    )
    model.load_model(args.model_path)     # DDP-safe inside TrajectoryModel
    core = model.model                    # underlying nn.Module (possibly wrapped later)

    # Wrap with DDP for parallel inference (optional but helps throughput)
    if world_size() > 1:
        core.to(device)
        model.model = DDP(core, device_ids=[local_rank], output_device=local_rank)

    # -----------------------------
    # 3) Evaluate on each rank
    # -----------------------------
    test_loss, test_acc = model.evaluate(test_loader)
    test_loss = reduce_mean_scalar(test_loss, device)
    test_acc  = reduce_mean_scalar(test_acc,  device)

    # Gather labels/preds from all ranks for global report
    # We'll reuse model.predict to get local shard outputs first.
    local_labels, local_preds = model.predict(test_loader)

    # Use all_gather_object for variable-length Python lists
    gathered_labels = [None for _ in range(world_size())]
    gathered_preds  = [None for _ in range(world_size())]
    if world_size() > 1:
        dist.all_gather_object(gathered_labels, local_labels)
        dist.all_gather_object(gathered_preds,  local_preds)
        if is_primary():
            all_labels = []
            all_preds  = []
            for lab, pr in zip(gathered_labels, gathered_preds):
                all_labels.extend(lab)
                all_preds.extend(pr)
    else:
        all_labels = local_labels
        all_preds  = local_preds

    # -----------------------------
    # 4) Rank-0 reporting/plots
    # -----------------------------
    if is_primary():
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        report = classification_report(all_labels, all_preds, target_names=processor.label_encoder.classes_)
        print("Classification Report:\n", report)

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=processor.label_encoder.classes_,
                    yticklabels=processor.label_encoder.classes_)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix (DDP Test)')
        plt.tight_layout()
        plt.savefig('confusion_matrix_test.png', dpi=200)

    ddp_cleanup()


if __name__ == "__main__":
    main()