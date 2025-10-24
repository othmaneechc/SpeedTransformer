#!/usr/bin/env python3
"""
Rockfish driver for SpeedTransformer & LSTM runs, attention visualization, and resource monitoring.

- Trains LSTM/Transformer (single- or multi-GPU via torchrun)
- Visualizes attention weights from a saved Transformer checkpoint (state_dict or full module)
- Monitors CPU/GPU utilization and memory; saves CSV + summary plots (300 dpi)

Examples (inside a Slurm job):
  python run_speedtransformer_rf.py --root /path/to/SpeedTransformer --do-transformer --do-lstm --do-plot --monitor

  # Visualize attention weights from a checkpoint (edit class-name to your model)
  python run_speedtransformer_rf.py \
    --root /path/to/SpeedTransformer \
    --viz-attn \
    --model-path "/home/ext-cchang/scr4_yang1/ext-cchang/SpeedTransformer/experiments/transformer_sweeps/lr1e-4_bs512_h8_d128_kv4_do0.1/best_model.pth" \
    --module-path "/home/ext-cchang/scr4_yang1/ext-cchang/SpeedTransformer" \
    --class-name "TrajectoryTransformer" \
    --init-kwargs '{"d_model":128,"nhead":8,"kv_heads":4,"pre_ln":true,"use_rope":true}' \
    --batch-first --device cuda --monitor
"""

import argparse
import os
import sys
import subprocess
import time
import threading
import json
import re
from pathlib import Path
from datetime import datetime

# headless plotting for batch jobs
os.environ.setdefault("MPLBACKEND", "Agg")

# ROOT will be set from command line argument
ROOT = None
LSTM_DIR = None
TRANS_DIR = None
DATA_DIR = None
OUT_DIR = None

# Expected data files (will be initialized after ROOT is set)
GEOLIFE = None
MOBIS = None  
MINIPROG = None

def initialize_data_paths():
    """Initialize data file paths after ROOT is set"""
    global GEOLIFE, MOBIS, MINIPROG
    GEOLIFE  = DATA_DIR / "geolife_processed.csv"
    MOBIS    = DATA_DIR / "mobis_processed.csv"
    MINIPROG = DATA_DIR / "miniprogram_balanced.csv"

# ------------------------- helpers -------------------------

def gpu_count_from_env() -> int:
    # explicit override
    n_override = os.environ.get("TORCH_NPROC_PER_NODE")
    if n_override and n_override.isdigit():
        return max(1, int(n_override))

    # SLURM hint (e.g., "gpu:a100:4" or "4")
    slurm = os.environ.get("SLURM_GPUS_PER_NODE")
    if slurm:
        try:
            return max(1, int(str(slurm).split(":")[-1]))
        except Exception:
            pass

    # CUDA_VISIBLE_DEVICES
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        try:
            return max(1, len([x for x in cvd.split(",") if x != ""]))
        except Exception:
            pass

    return 1


def run(cmd: str, cwd: Path = None):
    print(f"\n[RUN] {cmd}\n[ CWD ] {cwd or Path.cwd()}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=str(cwd) if cwd else None)


def ddp_or_python(exe: str, args: str, workdir: Path):
    """
    Launch `exe` either with torchrun (multi-GPU) or python (single).
    `exe` is the script filename inside `workdir`.
    """
    nproc = gpu_count_from_env()
    if nproc > 1:
        cmd = f"torchrun --standalone --nproc_per_node={nproc} {exe} {args}"
    else:
        cmd = f"python {exe} {args}"
    run(cmd, cwd=workdir)


def check_exists(p: Path, what: str) -> bool:
    if not p.exists():
        print(f"[WARN] Missing {what}: {p}")
        return False
    return True

# ------------------------- resource monitor -------------------------

class ResourceMonitor:
    """
    Background sampler for CPU%, RAM, GPU util/mem using psutil + nvidia-smi.
    Writes CSV and plots to OUT_DIR / {tag}_resources.* (300 dpi).
    """
    def __init__(self, tag: str, interval: float = 2.0):
        self.tag = tag
        self.interval = interval
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.csv_path = OUT_DIR / f"{tag}_resources.csv"
        self.start_time = None

        try:
            import psutil  # noqa
            self._has_psutil = True
        except Exception:
            print("[MON] psutil not available; CPU/RAM monitoring disabled.")
            self._has_psutil = False

        # Test nvidia-smi availability
        try:
            subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._has_nvsmi = True
        except Exception:
            print("[MON] nvidia-smi not available; GPU monitoring disabled.")
            self._has_nvsmi = False

    def _gpu_stats(self):
        if not self._has_nvsmi:
            return []
        q = "--query-gpu=index,utilization.gpu,utilization.memory,memory.total,memory.used"
        p = subprocess.run(["nvidia-smi", q, "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines = [ln.strip() for ln in p.stdout.strip().splitlines() if ln.strip()]
        stats = []
        for ln in lines:
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) >= 5:
                gpu_idx, util, mem_util, mem_total, mem_used = parts[:5]
                stats.append({
                    "gpu": int(gpu_idx),
                    "gpu_util": float(util),
                    "mem_util": float(mem_util),
                    "mem_total": float(mem_total),
                    "mem_used": float(mem_used),
                })
        return stats

    def _loop(self):
        if self._has_psutil:
            import psutil  # local import
        self.start_time = time.time()
        with open(self.csv_path, "w") as f:
            f.write("t_sec,cpu_percent,ram_gb," +
                    "gpu_idx,gpu_util,mem_util,mem_total_mb,mem_used_mb\n")
        while not self._stop.is_set():
            t = time.time() - self.start_time
            cpu = ram_gb = None
            if self._has_psutil:
                try:
                    import psutil
                    cpu = psutil.cpu_percent(interval=None)
                    ram_gb = psutil.virtual_memory().used / (1024**3)
                except Exception:
                    cpu = ram_gb = None

            gstats = self._gpu_stats() or [{"gpu": -1, "gpu_util": -1, "mem_util": -1, "mem_total": -1, "mem_used": -1}]
            with open(self.csv_path, "a") as f:
                for gs in gstats:
                    f.write("{:.1f},{:.1f},{:.3f},{},{:.1f},{:.1f},{:.1f},{:.1f}\n".format(
                        t, cpu if cpu is not None else -1.0,
                        ram_gb if ram_gb is not None else -1.0,
                        gs["gpu"], gs["gpu_util"], gs["mem_util"],
                        gs["mem_total"], gs["mem_used"]
                    ))
            time.sleep(self.interval)

    def start(self):
        print(f"[MON] Resource monitoring started: {self.csv_path}")
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=5)
        print(f"[MON] Resource monitoring stopped.")

    def summarize(self):
        """
        Create line plots for CPU, RAM, and per-GPU utilization/memory (300 dpi).
        """
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[MON] Skipping plots (pandas/matplotlib missing): {e}")
            return
        if not self.csv_path.exists():
            return
        df = pd.read_csv(self.csv_path)

        # CPU / RAM
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(df["t_sec"], df["cpu_percent"])
        ax[0].set_ylabel("CPU %")
        ax[0].grid(True, ls="--", alpha=0.4)
        ax[1].plot(df["t_sec"], df["ram_gb"])
        ax[1].set_ylabel("RAM (GB)")
        ax[1].set_xlabel("Time (s)")
        ax[1].grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"{self.tag}_cpu_ram.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # GPU util per index
        for gpu_idx in sorted(df["gpu_idx"].unique()):
            if gpu_idx < 0:
                continue
            g = df[df["gpu_idx"] == gpu_idx]
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax[0].plot(g["t_sec"], g["gpu_util"])
            ax[0].set_ylabel(f"GPU{gpu_idx} Util %")
            ax[0].grid(True, ls="--", alpha=0.4)
            ax[1].plot(g["t_sec"], g["mem_used_mb"])
            ax[1].set_ylabel(f"GPU{gpu_idx} Mem (MB)")
            ax[1].set_xlabel("Time (s)")
            ax[1].grid(True, ls="--", alpha=0.4)
            plt.tight_layout()
            fig.savefig(OUT_DIR / f"{self.tag}_gpu{gpu_idx}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

# ------------------------- training pipelines -------------------------

def train_lstm():
    ok1 = check_exists(GEOLIFE, "GeoLife data")
    ok2 = check_exists(MOBIS, "MOBIS data")
    if not (ok1 and ok2):
        print("[SKIP] LSTM training skipped due to missing data.")
        return
    ddp_or_python("lstm.py", f"--data_path {GEOLIFE} --random_state 1", LSTM_DIR)
    ddp_or_python("lstm.py", f"--data_path {MOBIS}   --random_state 316", LSTM_DIR)


def train_transformer():
    ok1 = check_exists(GEOLIFE, "GeoLife data")
    ok2 = check_exists(MOBIS, "MOBIS data")
    if not (ok1 and ok2):
        print("[SKIP] Transformer training skipped due to missing data.")
        return
    ddp_or_python("train.py", f"--data_path {MOBIS}   --random_state 1",   TRANS_DIR)
    ddp_or_python("train.py", f"--data_path {GEOLIFE} --random_state 316", TRANS_DIR)

# ------------------------- validation curves (300 dpi) -------------------------

def extract_metrics(file_path, patterns):
    with open(file_path, "r") as f:
        log_content = f.read()
    train_losses = [float(x) for x in re.findall(patterns["train_loss"], log_content)]
    train_accs   = [float(x) for x in re.findall(patterns["train_acc"],   log_content)]
    val_losses   = [float(x) for x in re.findall(patterns["val_loss"],    log_content)]
    val_accs     = [float(x) for x in re.findall(patterns["val_acc"],     log_content)]
    return train_losses, train_accs, val_losses, val_accs


def plot_val_comparison():
    import matplotlib.pyplot as plt

    files = {
        "Geolife LSTM":        LSTM_DIR  / "geolife" / "lstm.log",
        "Geolife Transformer": TRANS_DIR / "geolife" / "train.log",
        "Mobis LSTM":          LSTM_DIR  / "mobis"   / "lstm.log",
        "Mobis Transformer":   TRANS_DIR / "mobis"   / "train.log",
    }
    lstm_patterns = {
        "train_loss": r"Train Loss: ([\d.]+)",
        "train_acc":  r"Train Acc: ([\d.]+)",
        "val_loss":   r"Val Loss: ([\d.]+)",
        "val_acc":    r"Val Acc: ([\d.]+)",
    }
    trans_patterns = {
        "train_loss": r"Train Loss: ([\d.]+)",
        "train_acc":  r"Train Acc: ([\d.]+)",
        "val_loss":   r", Val Loss: ([\d.]+)",
        "val_acc":    r"Val Acc: ([\d.]+)",
    }

    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        print(f"[WARN] Missing log files for: {missing}. Skipping plot.")
        return

    gl_lstm = extract_metrics(files["Geolife LSTM"], lstm_patterns)
    gl_tr   = extract_metrics(files["Geolife Transformer"], trans_patterns)
    mb_lstm = extract_metrics(files["Mobis LSTM"], lstm_patterns)
    mb_tr   = extract_metrics(files["Mobis Transformer"], trans_patterns)

    def epochs(v): return list(range(1, len(v) + 1))

    geolife_data = (epochs(gl_lstm[3]), gl_lstm[3], epochs(gl_tr[3]), gl_tr[3])
    mobis_data   = (epochs(mb_lstm[3]), mb_lstm[3], epochs(mb_tr[3]), mb_tr[3])

    plt.figure(figsize=(15, 5))

    # Geolife
    plt.subplot(1, 2, 1)
    e_lstm, acc_lstm, e_tr, acc_tr = geolife_data
    plt.plot(e_tr,  [v * 100 for v in acc_tr],  label="Transformer", linestyle="--", marker="o")
    plt.plot(e_lstm, acc_lstm,                  label="LSTM",        linestyle="-.", marker="s")
    plt.title("Validation Accuracy on Geolife")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # MOBIS
    plt.subplot(1, 2, 2)
    e_lstm, acc_lstm, e_tr, acc_tr = mobis_data
    plt.plot(e_tr,  [v * 100 for v in acc_tr],  label="Transformer", linestyle="--", marker="o")
    plt.plot(e_lstm, acc_lstm,                  label="LSTM",        linestyle="-.", marker="s")
    plt.title("Validation Accuracy on MOBIS")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    out = OUT_DIR / "val.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved plot to {out}")

# ------------------------- attention visualization (300 dpi) -------------------------

def _unwrap_state_dict(obj):
    import torch
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return None


def load_model_for_viz(model_path, module_path=None, class_name=None, device="cpu", init_kwargs=None):
    import torch, torch.nn as nn
    if module_path and class_name:
        sys.path.append(os.path.abspath(module_path))
        mod_name = Path(module_path).name
        user_mod = __import__(mod_name)
        if not hasattr(user_mod, class_name):
            raise RuntimeError(f"Module '{mod_name}' has no class '{class_name}'.")
        ModelClass = getattr(user_mod, class_name)
        ctor_kwargs = json.loads(init_kwargs) if init_kwargs else {}
        model = ModelClass(**ctor_kwargs)
        ckpt = torch.load(model_path, map_location=device)
        state = _unwrap_state_dict(ckpt)
        if state is None:
            raise RuntimeError("Checkpoint missing recognizable state_dict.")
        # strip DDP 'module.' prefixes
        new_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        model.load_state_dict(new_state, strict=False)
        model.to(device).eval()
        return model
    # Try full module
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, nn.Module):
        obj.eval().to(device)
        return obj
    raise RuntimeError("Provide --module-path and --class-name (checkpoint is not a full nn.Module).")


def patch_mha_need_weights(model):
    import torch.nn as nn, types
    for _, m in model.named_modules():
        if isinstance(m, nn.MultiheadAttention):
            orig_forward = m.forward
            def wrapped_forward(self, query, key, value, **kwargs):
                kwargs = dict(kwargs)
                kwargs["need_weights"] = True
                return orig_forward(query, key, value, **kwargs)
            m.forward = types.MethodType(wrapped_forward, m)


class AttnCollector:
    def __init__(self):
        self.collected = []
        self._handles = []
    def _hook(self, name):
        def fn(module, inputs, output):
            if isinstance(output, tuple) and len(output) == 2:
                attn_output, attn_weights = output
                self.collected.append({"name": name, "weights": attn_weights.detach().cpu()})
        return fn
    def register(self, model):
        import torch.nn as nn
        for name, m in model.named_modules():
            if isinstance(m, nn.MultiheadAttention):
                h = m.register_forward_hook(self._hook(name))
                self._handles.append(h)
    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def plot_heatmap(mat, title, outfile):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heads_grid(attn_bhqq, title, outfile, max_heads=16):
    import numpy as np, math
    import matplotlib.pyplot as plt
    B, H, Tq, Tk = attn_bhqq.shape
    H_show = min(H, max_heads)
    cols = int(math.ceil(math.sqrt(H_show)))
    rows = int(math.ceil(H_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    axes = np.array(axes).reshape(rows, cols)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if idx < H_show:
                head_map = attn_bhqq[0, idx].numpy()
                im = ax.imshow(head_map, aspect="auto")
                ax.set_title(f"Head {idx}")
                ax.set_xlabel("Key"); ax.set_ylabel("Query")
                ax.axis("on")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                idx += 1
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_attention(model_path, module_path, class_name, init_kwargs,
                        device, batch_first, seq_len, d_model, input_npy,
                        save_tag="attn_viz"):
    import numpy as np, torch
    save_dir = OUT_DIR / save_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_for_viz(model_path, module_path, class_name, device, init_kwargs)
    patch_mha_need_weights(model)
    collector = AttnCollector()
    collector.register(model)

    # Build input
    if input_npy:
        arr = np.load(input_npy)
        if arr.ndim == 2:
            arr = arr[None, ...]
        x = torch.tensor(arr, dtype=torch.float32)
    else:
        B = 1
        x = torch.randn(B, seq_len, d_model) if batch_first else torch.randn(seq_len, B, d_model)
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        try:
            _ = model(x)
        except TypeError:
            try:
                _ = model(x, None)
            except Exception as e:
                raise RuntimeError("Model forward() failed; adapt inputs/masks for your model.") from e

    if not collector.collected:
        print("[ATTN] No attention weights captured (non-MHA attention or custom module?).")
    else:
        for idx, item in enumerate(collector.collected):
            name = item["name"].replace(".", "_")
            w = item["weights"]  # [B, H, Tq, Tk]
            avg = w.mean(dim=1)[0].numpy()
            plot_heatmap(avg, title=f"{name} — Avg heads", outfile=save_dir / f"{idx:02d}_{name}_avg.png")
            plot_heads_grid(w, title=f"{name} — Per-head", outfile=save_dir / f"{idx:02d}_{name}_heads.png")
        print(f"[ATTN] Saved attention visualizations to: {save_dir}")
    collector.remove()

# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()
    # root directory
    parser.add_argument("--root", type=str, required=True, help="Root directory path for SpeedTransformer project")
    # training/plots
    parser.add_argument("--do-lstm", action="store_true", help="Run LSTM trainings")
    parser.add_argument("--do-transformer", action="store_true", help="Run Transformer trainings")
    parser.add_argument("--do-plot", action="store_true", help="Make validation comparison figure")
    # attention viz
    parser.add_argument("--viz-attn", action="store_true", help="Visualize attention from a saved Transformer")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--module-path", type=str, default=None, help="Directory with model class file")
    parser.add_argument("--class-name", type=str, default=None, help="Model class name")
    parser.add_argument("--init-kwargs", type=str, default=None, help='JSON ctor kwargs e.g. {"d_model":128,"nhead":8}')
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--batch-first", action="store_true", help="Model expects [B,T,d]")
    parser.add_argument("--seq-len", type=int, default=200, help="Dummy input length if no npy provided")
    parser.add_argument("--d-model", type=int, default=128, help="Dummy input dim if no npy provided")
    parser.add_argument("--input-npy", type=str, default=None, help="Optional npy input [B,T,d] or [T,d]")
    parser.add_argument("--save-tag", type=str, default=None, help="Folder tag for outputs in outputs/")
    # monitoring
    parser.add_argument("--monitor", action="store_true", help="Record CPU/GPU usage during run")
    parser.add_argument("--sample-interval", type=float, default=2.0, help="Monitor sampling interval (s)")
    args = parser.parse_args()

    # Initialize ROOT and derived paths from command line argument
    global ROOT, LSTM_DIR, TRANS_DIR, DATA_DIR, OUT_DIR
    ROOT = Path(args.root).resolve()
    LSTM_DIR = ROOT / "models" / "lstm"           # change if your lstm.py is elsewhere
    TRANS_DIR = ROOT                               # <-- train.py is at project root
    DATA_DIR  = ROOT / "data"
    OUT_DIR   = ROOT / "outputs"

    # Ensure dirs
    for d in [LSTM_DIR, TRANS_DIR, DATA_DIR, OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialize data file paths
    initialize_data_paths()

    print(f"[INFO] ROOT={ROOT}")
    print(f"[INFO] GPUs visible/heuristic: {gpu_count_from_env()}")

    tag = args.save_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    mon = None
    if args.monitor:
        mon = ResourceMonitor(tag=tag, interval=args.sample_interval)
        mon.start()

    t0 = time.time()
    try:
        if args.do_lstm:
            train_lstm()

        if args.do_transformer:
            train_transformer()

        if args.do_plot:
            plot_val_comparison()

        if args.viz_attn:
            if not args.model_path:
                raise ValueError("--model-path is required for --viz-attn")
            visualize_attention(
                model_path=args.model_path,
                module_path=args.module_path,
                class_name=args.class_name,
                init_kwargs=args.init_kwargs,
                device=args.device,
                batch_first=args.batch_first,
                seq_len=args.seq_len,
                d_model=args.d_model,
                input_npy=args.input_npy,
                save_tag=f"{tag}_attn"
            )

    finally:
        if mon:
            mon.stop()
            mon.summarize()
        dt = time.time() - t0
        print(f"[DONE] Total elapsed: {dt:.1f}s ; outputs in {OUT_DIR}")

if __name__ == "__main__":
    sys.exit(main())