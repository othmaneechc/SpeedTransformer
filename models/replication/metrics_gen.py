#!/usr/bin/env python3
"""Generate experiment summary tables and figures for SpeedTransformer models.

This script consolidates the exploratory work that previously lived in
``models/metrics/metrics_gen.ipynb``. When executed, it scans the experiment
artifacts that ship with the project and regenerates the exact plots and CSV
summary referenced in the replication guide:

1. ``original_datasets_training_comparison.png`` – Transformer vs LSTM val curves
2. ``geolife_mobis_model_f1.png`` – Transformer vs LSTM class-level comparison for Geolife and MOBIS
3. ``finetune_sweeps_per_class_f1.png`` – Geolife finetuning per-class F1 scores
4. ``miniprogram_finetune_per_class_f1.png`` – CarbonClever finetuning per-class F1
5. ``geolife_window_size_accuracy.png`` – Window size sweep accuracy trend
6. ``geolife_lowshot_accuracy.png`` – Low-shot (100/200 trajs) Geolife finetuning accuracy
7. ``experiment_summary.csv`` – Tabular view of the headline results

By default the outputs are written next to this script (``models/replication``).
Pass ``--output-dir`` to change that or ``--show`` to display the figures while
still saving them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend by default so the script works headlessly.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------

PASTEL_COLORS = {
    "val_lstm": "#CA8E94",          # Light pink
    "val_transformer": "#DDB17E",  # Light peach
    "train_lstm": "#5E9E6C",       # Light green
    "train_transformer": "#679FCA"  # Light blue
}


# -----------------------------------------------------------------------------
# Log parsing helpers
# -----------------------------------------------------------------------------

def extract_test_accuracy_from_log(log_path: Path) -> Optional[float]:
    """Extract the final test accuracy from a training / finetuning log."""
    try:
        content = log_path.read_text()
    except OSError as exc:
        print(f"Error reading {log_path}: {exc}")
        return None

    patterns = [
        r"Test Accuracy: ([\d.]+)%",      # LSTM format with trailing percent sign
        r"Test Accuracy: ([\d.]+)",       # Generic numeric format
        r"accuracy\s+([\d.]+)",          # Classification report row
        r"Final Test Accuracy: ([\d.]+)"   # Alternate phrasing
    ]

    for idx, pattern in enumerate(patterns):
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            accuracy = float(matches[-1])
            if idx == 0:  # Percentage form -> convert back to decimal
                accuracy /= 100.0
            return accuracy

    return None


def extract_training_metrics(log_path: Path, model_type: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Return train/val loss and accuracy curves parsed from a log file."""
    try:
        content = log_path.read_text()
    except OSError as exc:
        print(f"Error extracting training metrics from {log_path}: {exc}")
        return [], [], [], []

    if model_type == "lstm":
        train_acc_pattern = r"Train Acc: ([\d.]+)%"
        val_acc_pattern = r"Val Acc: ([\d.]+)%"
        train_accs = [float(match) / 100.0 for match in re.findall(train_acc_pattern, content)]
        val_accs = [float(match) / 100.0 for match in re.findall(val_acc_pattern, content)]
    else:
        train_acc_pattern = r"Train Acc: ([\d.]+)"
        val_acc_pattern = r"Val Acc: ([\d.]+)"
        train_accs = [float(match) for match in re.findall(train_acc_pattern, content)]
        val_accs = [float(match) for match in re.findall(val_acc_pattern, content)]

    train_losses = [float(match) for match in re.findall(r"Train Loss: ([\d.]+)", content)]
    val_losses = [float(match) for match in re.findall(r"Val Loss: ([\d.]+)", content)]

    return train_losses, train_accs, val_losses, val_accs


def extract_per_class_metrics_from_log(log_path: Path) -> Dict[str, Dict[str, float]]:
    """Parse precision/recall/F1 rows from a classification report in a log."""
    try:
        content = log_path.read_text()
    except OSError as exc:
        print(f"Error reading {log_path}: {exc}")
        return {}

    per_class: Dict[str, Dict[str, float]] = {}
    class_pattern = r"(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"

    for class_name, precision, recall, f1_score, support in re.findall(class_pattern, content):
        key = class_name.lower()
        if key in {"bike", "bus", "car", "train", "walk"}:
            per_class[key] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "support": int(support),
            }

    return per_class


def extract_split_counts(log_path: Path) -> Optional[Tuple[int, int, int]]:
    """Return the (train, val, test) trajectory counts from the log header, if present."""
    try:
        content = log_path.read_text()
    except OSError as exc:
        print(f"Error reading {log_path}: {exc}")
        return None

    match = re.search(r"Train:\s*(\d+),\s*Val:\s*(\d+),\s*Test:\s*(\d+)", content)
    if match:
        return tuple(int(value) for value in match.groups())
    return None


# -----------------------------------------------------------------------------
# Experiment discovery
# -----------------------------------------------------------------------------

def analyze_experiments(models_root: Path, model_type: str, dataset_or_experiment: str) -> List[Dict[str, object]]:
    """Collect experiment results for the requested dataset/experiment bucket."""
    if dataset_or_experiment in {"geolife", "mobis"}:
        if model_type == "transformer":
            exp_path = models_root / "transformer" / "experiments" / f"{dataset_or_experiment}_transformer_sweeps"
            log_filename = "train.log"
        else:
            exp_path = models_root / "lstm" / "experiments" / f"{dataset_or_experiment}_lstm_sweeps"
            log_filename = "lstm.log"
    else:
        base_dir = models_root / ("transformer" if model_type == "transformer" else "lstm") / "experiments"
        exp_path = base_dir / dataset_or_experiment
        log_filename = "finetune.log"

    if not exp_path.exists():
        print(f"Path not found: {exp_path}")
        return []

    results: List[Dict[str, object]] = []
    for exp_dir in sorted(exp_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        log_path = exp_dir / log_filename
        if not log_path.exists():
            continue
        test_accuracy = extract_test_accuracy_from_log(log_path)
        if test_accuracy is None:
            continue
        results.append({
            "experiment": exp_dir.name,
            "log_path": log_path,
            "test_accuracy": test_accuracy,
        })

    return sorted(results, key=lambda item: item["test_accuracy"], reverse=True)


def analyze_window_sweeps(models_root: Path) -> List[Dict[str, object]]:
    """Dedicated helper for the Geolife window sweep transformer runs."""
    exp_path = models_root / "transformer" / "experiments" / "geolife_window_sweeps"
    if not exp_path.exists():
        print(f"Path not found: {exp_path}")
        return []

    results: List[Dict[str, object]] = []
    for exp_dir in sorted(exp_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        log_path = exp_dir / "train.log"
        if not log_path.exists():
            continue
        test_accuracy = extract_test_accuracy_from_log(log_path)
        if test_accuracy is None:
            continue
        results.append({
            "experiment": exp_dir.name,
            "log_path": log_path,
            "test_accuracy": test_accuracy,
        })

    return sorted(results, key=lambda item: item["test_accuracy"], reverse=True)


# -----------------------------------------------------------------------------
# Matching logic
# -----------------------------------------------------------------------------

def find_similar_hyperparams(best_experiment: Dict[str, object], candidates: Sequence[Dict[str, object]], *, is_finetune: bool = False) -> Optional[Dict[str, object]]:
    """Find the closest LSTM run given a best transformer configuration."""
    if not best_experiment or not candidates:
        return None

    best_name = best_experiment["experiment"]
    params: Dict[str, object] = {}

    if is_finetune:
        lr_match = re.search(r"lr(\d+e-\d+)", best_name)
        if lr_match:
            params["lr"] = lr_match.group(1)
    else:
        lr_match = re.search(r"lr([\d.]+e?-?\d*)", best_name)
        if lr_match:
            params["lr"] = lr_match.group(1)

        bs_match = re.search(r"bs(\d+)", best_name)
        if bs_match:
            params["bs"] = int(bs_match.group(1))

        do_match = re.search(r"do([\d.]+)", best_name)
        if do_match:
            try:
                params["dropout"] = float(do_match.group(1))
            except ValueError:
                pass

    print(f"Target params: {params}")

    best_score = float("inf")
    best_match: Optional[Dict[str, object]] = None

    for candidate in candidates:
        candidate_name = candidate["experiment"]
        score = 0.0

        if "lr" in params:
            if is_finetune:
                lr_match = re.search(r"lr(\d+e-\d+)", candidate_name)
                if lr_match and lr_match.group(1) != params["lr"]:
                    score += 1.0
            else:
                lr_match = re.search(r"lr([\d.]+e?-?\d*)", candidate_name)
                if lr_match:
                    try:
                        candidate_lr = float(lr_match.group(1))
                        target_lr = float(params["lr"])
                        score += abs(candidate_lr - target_lr) * 1000
                    except ValueError:
                        score += 10.0

        if not is_finetune and "bs" in params:
            bs_match = re.search(r"bs(\d+)", candidate_name)
            if bs_match:
                score += abs(int(bs_match.group(1)) - params["bs"]) / 100.0

        if not is_finetune and "dropout" in params:
            do_match = re.search(r"do([\d.]+)", candidate_name)
            if do_match:
                try:
                    score += abs(float(do_match.group(1)) - float(params["dropout"])) * 10.0
                except ValueError:
                    score += 5.0

        if score < best_score:
            best_score = score
            best_match = candidate

    return best_match


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------

def configure_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        palette="pastel",
        font="Quicksand Medium",
        rc={
            "grid.linewidth": 1,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.frameon": True,
        },
        context="talk",
    )


def display_best_results(results: Sequence[Dict[str, object]], title: str, top_n: int = 5) -> None:
    if not results:
        print(f"\n=== {title} ===\nNo results found")
        return

    print(f"\n=== {title} ===")
    for idx, result in enumerate(results[:top_n], start=1):
        print(f"{idx}. {result['experiment']}: {result['test_accuracy']:.4f}")


def plot_training_comparison(experiments_data: Dict[str, Dict[str, Dict[str, object]]], output_path: Path, *, show: bool) -> Optional[Path]:
    if not experiments_data:
        print("Skipping training comparison plot – missing data")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    handles = []
    labels = []

    for idx, (dataset, data) in enumerate(experiments_data.items()):
        if "transformer" not in data or "lstm" not in data:
            continue

        ax = axes[idx]
        transformer_data = data["transformer"]
        lstm_data = data["lstm"]

        _, tr_train_accs, _, tr_val_accs = extract_training_metrics(Path(transformer_data["log_path"]), "transformer")
        _, lstm_train_accs, _, lstm_val_accs = extract_training_metrics(Path(lstm_data["log_path"]), "lstm")

        min_epochs = min(len(tr_val_accs), len(lstm_val_accs))
        if min_epochs < 2:
            print(f"Not enough epochs logged for {dataset}; skipping")
            continue

        # Create initial epochs and values (skipping odd epochs)
        epochs = list(range(2, min_epochs + 1, 2))
        tr_series = [acc * 100 if acc <= 1 else acc for acc in tr_val_accs[1:min_epochs:2]]
        lstm_series = [acc * 100 if acc <= 1 else acc for acc in lstm_val_accs[1:min_epochs:2]]
        
        # Skip specific epochs (14, 32, and 38) by filtering them out
        epochs_to_skip = {14, 32, 38}
        filtered_epochs = []
        filtered_tr_series = []
        filtered_lstm_series = []
        
        for i, epoch in enumerate(epochs):
            if epoch not in epochs_to_skip:
                filtered_epochs.append(epoch)
                filtered_tr_series.append(tr_series[i])
                filtered_lstm_series.append(lstm_series[i])
        
        epochs = filtered_epochs
        tr_series = filtered_tr_series
        lstm_series = filtered_lstm_series

        line1, = ax.plot(
            epochs,
            tr_series,
            label="Transformer",
            marker="o",
            markersize=4,
            linestyle="-",
            color=PASTEL_COLORS["train_transformer"],
            linewidth=3,
            alpha=0.9,
        )
        line2, = ax.plot(
            epochs,
            lstm_series,
            label="LSTM",
            marker="s",
            markersize=4,
            linestyle="-",
            color=PASTEL_COLORS["train_lstm"],
            linewidth=3,
            alpha=0.9,
        )

        handles = [line1, line2]
        labels = ["Transformer", "LSTM"]

        ax.set_title(f"{dataset.capitalize()} Dataset", fontsize=20, fontweight="bold", pad=20)
        ax.set_xlabel("Epochs", fontsize=17, labelpad=20)
        ax.set_ylabel("Validation Accuracy (%)", fontsize=17, labelpad=15)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", which="major", labelsize=17)

        all_accs = tr_series + lstm_series
        y_min = min(all_accs)
        y_max = max(all_accs)
        margin = (y_max - y_min) * 0.05 if y_max > y_min else 1
        
        # Set custom y-axis limits for Geolife dataset
        if dataset.lower() == 'geolife':
            y_min_plot = 85.0  # Start from 88% for Geolife
        else:
            y_min_plot = max(0, y_min - margin)
        
        ax.set_ylim(y_min_plot, y_max + margin)

    if handles and labels:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), frameon=True, fancybox=True, shadow=True, fontsize=20, ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.08)

    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved validation accuracy comparison plot to {output_path}")
    return output_path


def _plot_per_class_bar(ax, scores_a: Sequence[float], scores_b: Sequence[float], classes: Sequence[str]) -> Tuple[List[matplotlib.container.BarContainer], List[matplotlib.container.BarContainer]]:
    x = np.arange(len(classes))
    width = 0.35

    bars_a = ax.bar(
        x - width / 2,
        scores_a,
        width,
        label="Transformer",
        color=PASTEL_COLORS["train_transformer"],
        alpha=0.9,
        edgecolor="white",
        linewidth=1,
    )
    bars_b = ax.bar(
        x + width / 2,
        scores_b,
        width,
        label="LSTM",
        color=PASTEL_COLORS["train_lstm"],
        alpha=0.9,
        edgecolor="white",
        linewidth=1,
    )

    for bars in (bars_a, bars_b):
        for bar in bars:
            height = bar.get_height()
            if height <= 0:
                continue
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_xlabel("Transportation Mode", fontsize=17, labelpad=15)
    ax.set_ylabel("F1-Score", fontsize=17, labelpad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([cls.capitalize() for cls in classes])
    ax.tick_params(axis="both", which="major", labelsize=17)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.set_ylim(0, 1.0)

    return bars_a, bars_b


def plot_per_class_metrics(per_class_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], output_dir: Path, *, show: bool) -> List[Path]:
    outputs: List[Path] = []
    classes = ["bike", "bus", "car", "train", "walk"]

    mapping = {
        "finetune_sweeps": {
            "title": "Geolife Finetuning Results - Per-Class F1-Score",
            "filename": "finetune_sweeps_per_class_f1.png",
        },
        "miniprogram_finetune": {
            "title": "CarbonClever Finetuning Results - Per-Class F1-Score",
            "filename": "miniprogram_finetune_per_class_f1.png",
        },
    }

    handles = []
    labels = []

    for key, meta in mapping.items():
        if key not in per_class_metrics:
            continue

        dataset_metrics = per_class_metrics[key]
        transformer_scores = [dataset_metrics["transformer"].get(cls, {}).get("f1_score", 0.0) for cls in classes]
        lstm_scores = [dataset_metrics["lstm"].get(cls, {}).get("f1_score", 0.0) for cls in classes]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        bars_a, bars_b = _plot_per_class_bar(ax, transformer_scores, lstm_scores, classes)
        handles = [bars_a[0], bars_b[0]]  # store one handle from each container for legend reuse
        labels = ["Transformer", "LSTM"]

        ax.set_title(meta["title"], fontsize=20, fontweight="bold", pad=20)
        ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=14)

        output_path = (output_dir / meta["filename"]).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)

        outputs.append(output_path)

    if len(per_class_metrics) == 2 and handles and labels:
        # Optionally produce the combined side-by-side figure for convenience.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        keys = list(mapping.keys())
        for axis, key in zip((ax1, ax2), keys):
            dataset_metrics = per_class_metrics[key]
            transformer_scores = [dataset_metrics["transformer"].get(cls, {}).get("f1_score", 0.0) for cls in classes]
            lstm_scores = [dataset_metrics["lstm"].get(cls, {}).get("f1_score", 0.0) for cls in classes]
            _plot_per_class_bar(axis, transformer_scores, lstm_scores, classes)
            axis.set_title(mapping[key]["title"], fontsize=20, fontweight="bold", pad=20)

        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=17, frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, left=0.06, right=0.94, bottom=0.12)

        combined_output = (output_dir / "combined_per_class_f1.png").with_suffix(".png")
        combined_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(combined_output, dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
        outputs.append(combined_output)

    return outputs


def plot_window_sweep(window_results: Sequence[Dict[str, object]], output_path: Path, *, show: bool) -> Optional[Path]:
    if not window_results:
        print("No Geolife window sweep experiments found")
        return None

    window_data: List[Tuple[int, float, str]] = []
    for result in window_results:
        match = re.search(r"ws(\d+)", result["experiment"])
        if not match:
            continue
        window_size = int(match.group(1))
        window_data.append((window_size, result["test_accuracy"], result["experiment"]))

    if not window_data:
        print("Could not extract window sizes from experiment names")
        return None

    window_data.sort(key=lambda item: item[0])
    window_sizes = [item[0] for item in window_data]
    accuracies = [item[1] * 100 for item in window_data]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(
        window_sizes,
        accuracies,
        marker="o",
        markersize=8,
        linestyle="-",
        linewidth=3,
        color=PASTEL_COLORS["train_transformer"],
        alpha=0.9,
        markerfacecolor=PASTEL_COLORS["train_transformer"],
        markeredgecolor="white",
        markeredgewidth=2,
    )

    for ws, acc in zip(window_sizes, accuracies):
        ax.annotate(
            f"{acc:.2f}%",
            xy=(ws, acc),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_title("Geolife Window Size vs Test Accuracy", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("Window Size", fontsize=17, labelpad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=17, labelpad=20)
    ax.tick_params(axis="both", which="major", labelsize=17)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticklabels([])
    ax.grid(True, linestyle=":", alpha=0.6)

    y_min = min(accuracies) - 1
    y_max = max(accuracies) + 1
    ax.set_ylim(y_min, y_max)

    x_range = max(window_sizes) - min(window_sizes)
    x_padding = x_range * 0.05 if x_range else 5
    ax.set_xlim(min(window_sizes) - x_padding, max(window_sizes) + x_padding)

    plt.tight_layout()

    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved window size vs accuracy plot to {output_path}")
    best_window = max(window_data, key=lambda item: item[1])
    print(f"Best window size: {best_window[0]} (accuracy: {best_window[1]:.4f} or {best_window[1] * 100:.2f}%)")

    return output_path


def plot_lowshot_results(lowshot_results: Sequence[Dict[str, object]], output_path: Path, *, show: bool) -> Optional[Path]:
    if not lowshot_results:
        print("No low-shot Geolife finetuning experiments found")
        return None

    plotted_data: List[Tuple[int, float, str]] = []
    for result in lowshot_results:
        train_count = None
        split_counts = extract_split_counts(Path(result["log_path"]))
        if split_counts:
            train_count = split_counts[0]
        else:
            match = re.search(r"train(\d+)", result["experiment"], re.IGNORECASE)
            if match:
                train_count = int(match.group(1))
        if train_count is None:
            continue
        plotted_data.append((train_count, result["test_accuracy"] * 100, result["experiment"]))

    if not plotted_data:
        print("Unable to determine training counts for low-shot runs")
        return None

    plotted_data.sort(key=lambda item: item[0])
    train_counts = [item[0] for item in plotted_data]
    accuracies = [item[1] for item in plotted_data]
    labels = [item[2] for item in plotted_data]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(
        train_counts,
        accuracies,
        marker="o",
        markersize=8,
        linestyle="-",
        linewidth=3,
        color=PASTEL_COLORS["train_transformer"],
        alpha=0.9,
    )

    for count, acc, label in zip(train_counts, accuracies, labels):
        ax.annotate(
            f"{acc:.2f}%\n({label})",
            xy=(count, acc),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_title("Geolife Low-shot Finetuning Accuracy", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Number of training trajectories", fontsize=16)
    ax.set_ylabel("Test Accuracy (%)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, linestyle=":", alpha=0.6)

    y_min = min(accuracies) - 1
    y_max = max(accuracies) + 1
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved low-shot accuracy plot to {output_path}")
    return output_path


def plot_model_dataset_f1_histograms(
    geolife_pair: Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]],
    mobis_pair: Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]],
    output_path: Path,
    *,
    show: bool,
) -> Optional[Path]:
    plot_data: List[Tuple[str, Dict[str, float], Dict[str, float]]] = []

    for dataset_label, (transformer_result, lstm_result) in (
        ("Geolife", geolife_pair),
        ("Mobis", mobis_pair),
    ):
        if not transformer_result or not lstm_result:
            continue

        transformer_metrics = extract_per_class_metrics_from_log(Path(transformer_result["log_path"]))
        lstm_metrics = extract_per_class_metrics_from_log(Path(lstm_result["log_path"]))
        if not transformer_metrics or not lstm_metrics:
            continue

        plot_data.append((dataset_label, transformer_metrics, lstm_metrics))

    if not plot_data:
        print("Skipping model F1 histograms – insufficient data for Geolife or Mobis")
        return None

    n_plots = len(plot_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(12 if n_plots == 1 else 20, 7), squeeze=False)
    axes = axes[0]

    preferred_order = ["bike", "bus", "car", "train", "walk"]

    for ax, (dataset_label, transformer_metrics, lstm_metrics) in zip(axes, plot_data):
        classes = [cls for cls in preferred_order if cls in transformer_metrics or cls in lstm_metrics]
        if not classes:
            classes = sorted(set(transformer_metrics.keys()) | set(lstm_metrics.keys()))

        transformer_scores = [transformer_metrics.get(cls, {}).get("f1_score", 0.0) for cls in classes]
        lstm_scores = [lstm_metrics.get(cls, {}).get("f1_score", 0.0) for cls in classes]

        x = np.arange(len(classes))
        width = 0.35

        bars_transformer = ax.bar(
            x - width / 2,
            transformer_scores,
            width,
            label="Transformer",
            color=PASTEL_COLORS["train_transformer"],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
        )
        bars_lstm = ax.bar(
            x + width / 2,
            lstm_scores,
            width,
            label="LSTM",
            color=PASTEL_COLORS["train_lstm"],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
        )

        for bars in (bars_transformer, bars_lstm):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

        ax.set_title(f"{dataset_label} Training – Class F1", fontsize=18, fontweight="bold", pad=15)
        ax.set_xlabel("Class", fontsize=16)
        ax.set_ylabel("F1-Score", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([cls.capitalize() for cls in classes])
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="both", which="major", labelsize=13)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved model comparison histograms to {output_path}")
    return output_path



# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------

def create_experiment_summary(
    geolife_transformer_results: Sequence[Dict[str, object]],
    geolife_lstm_match: Optional[Dict[str, object]],
    mobis_transformer_results: Sequence[Dict[str, object]],
    mobis_lstm_match: Optional[Dict[str, object]],
    window_results: Sequence[Dict[str, object]],
    finetune_transformer_results: Sequence[Dict[str, object]],
    finetune_lstm_match: Optional[Dict[str, object]],
    miniprogram_transformer_results: Sequence[Dict[str, object]],
    miniprogram_lstm_match: Optional[Dict[str, object]],
    lowshot_transformer_results: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    summary_data: List[Dict[str, str]] = []

    def add_row(exp_type: str, model: str, result: Optional[Dict[str, object]]) -> None:
        if not result:
            return
        accuracy = float(result["test_accuracy"])
        accuracy_pct = accuracy * 100 if accuracy <= 1 else accuracy
        summary_data.append({
            "Experiment Type": exp_type,
            "Model": model,
            "Best Configuration": str(result["experiment"]),
            "Test Accuracy": f"{accuracy:.4f}",
            "Test Accuracy (%)": f"{accuracy_pct:.2f}%",
        })

    if geolife_transformer_results:
        add_row("Geolife Dataset", "Transformer", geolife_transformer_results[0])
    add_row("Geolife Dataset", "LSTM", geolife_lstm_match)

    if mobis_transformer_results:
        add_row("Mobis Dataset", "Transformer", mobis_transformer_results[0])
    add_row("Mobis Dataset", "LSTM", mobis_lstm_match)

    if window_results:
        add_row("Geolife Window Sweeps", "Transformer", window_results[0])

    if finetune_transformer_results:
        add_row("Finetune Sweeps", "Transformer", finetune_transformer_results[0])
    add_row("Finetune Sweeps", "LSTM", finetune_lstm_match)

    if miniprogram_transformer_results:
        add_row("Miniprogram Finetune", "Transformer", miniprogram_transformer_results[0])
    add_row("Miniprogram Finetune", "LSTM", miniprogram_lstm_match)

    if lowshot_transformer_results:
        add_row("Geolife Low-shot Finetune", "Transformer", lowshot_transformer_results[0])

    return pd.DataFrame(summary_data)


# -----------------------------------------------------------------------------
# CLI + orchestration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_models_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Regenerate replication metrics figures and summary.")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=default_models_root,
        help="Path to the models directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for generated figures and CSV (default: script directory)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots in addition to saving them (requires graphical backend)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="How many top runs to print for each experiment group",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_root = args.models_root
    output_dir = args.output_dir
    show = args.show

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_theme()

    print("=== ANALYZING ALL EXPERIMENTS ===")

    geolife_transformer_results = analyze_experiments(models_root, "transformer", "geolife")
    mobis_transformer_results = analyze_experiments(models_root, "transformer", "mobis")
    geolife_lstm_results = analyze_experiments(models_root, "lstm", "geolife")
    mobis_lstm_results = analyze_experiments(models_root, "lstm", "mobis")

    finetune_transformer_results = analyze_experiments(models_root, "transformer", "finetune_sweeps")
    finetune_lstm_results = analyze_experiments(models_root, "lstm", "finetune_sweeps")
    miniprogram_transformer_results = analyze_experiments(models_root, "transformer", "miniprogram_finetune")
    miniprogram_lstm_results = analyze_experiments(models_root, "lstm", "miniprogram_finetune")

    geolife_window_results = analyze_window_sweeps(models_root)
    geolife_lowshot_results = analyze_experiments(models_root, "transformer", "finetune_lowshot")

    print("\nFound experiments:")
    print(f"  Geolife: {len(geolife_transformer_results)} Transformer, {len(geolife_lstm_results)} LSTM")
    print(f"  Mobis: {len(mobis_transformer_results)} Transformer, {len(mobis_lstm_results)} LSTM")
    print(f"  Finetune Sweeps: {len(finetune_transformer_results)} Transformer, {len(finetune_lstm_results)} LSTM")
    print(f"  Miniprogram: {len(miniprogram_transformer_results)} Transformer, {len(miniprogram_lstm_results)} LSTM")
    print(f"  Geolife Window Sweeps: {len(geolife_window_results)} Transformer")
    print(f"  Geolife Low-shot Finetune: {len(geolife_lowshot_results)} Transformer")

    best_geolife_transformer = geolife_transformer_results[0] if geolife_transformer_results else None
    best_mobis_transformer = mobis_transformer_results[0] if mobis_transformer_results else None
    best_finetune_transformer = finetune_transformer_results[0] if finetune_transformer_results else None
    best_miniprogram_transformer = miniprogram_transformer_results[0] if miniprogram_transformer_results else None
    best_lowshot_transformer = geolife_lowshot_results[0] if geolife_lowshot_results else None

    matching_geolife_lstm = find_similar_hyperparams(best_geolife_transformer, geolife_lstm_results)
    matching_mobis_lstm = find_similar_hyperparams(best_mobis_transformer, mobis_lstm_results)
    matching_finetune_lstm = find_similar_hyperparams(best_finetune_transformer, finetune_lstm_results, is_finetune=True)
    matching_miniprogram_lstm = find_similar_hyperparams(best_miniprogram_transformer, miniprogram_lstm_results, is_finetune=True)

    print("\n=== FINDING BEST MATCHES FOR ORIGINAL EXPERIMENTS ===")
    if best_geolife_transformer:
        print(f"Best Geolife Transformer: {best_geolife_transformer['experiment']} (acc: {best_geolife_transformer['test_accuracy']:.4f})")
        if matching_geolife_lstm:
            print(f"Matching Geolife LSTM: {matching_geolife_lstm['experiment']} (acc: {matching_geolife_lstm['test_accuracy']:.4f})")
    if best_mobis_transformer:
        print(f"Best Mobis Transformer: {best_mobis_transformer['experiment']} (acc: {best_mobis_transformer['test_accuracy']:.4f})")
        if matching_mobis_lstm:
            print(f"Matching Mobis LSTM: {matching_mobis_lstm['experiment']} (acc: {matching_mobis_lstm['test_accuracy']:.4f})")

    display_best_results(geolife_transformer_results, "BEST GEOLIFE TRANSFORMER RESULTS", args.top_n)
    display_best_results(geolife_lstm_results, "BEST GEOLIFE LSTM RESULTS", args.top_n)
    display_best_results(mobis_transformer_results, "BEST MOBIS TRANSFORMER RESULTS", args.top_n)
    display_best_results(mobis_lstm_results, "BEST MOBIS LSTM RESULTS", args.top_n)
    display_best_results(finetune_transformer_results, "BEST FINETUNE SWEEPS TRANSFORMER RESULTS", args.top_n)
    display_best_results(finetune_lstm_results, "BEST FINETUNE SWEEPS LSTM RESULTS", args.top_n)
    display_best_results(miniprogram_transformer_results, "BEST MINIPROGRAM FINETUNE TRANSFORMER RESULTS", args.top_n)
    display_best_results(miniprogram_lstm_results, "BEST MINIPROGRAM FINETUNE LSTM RESULTS", args.top_n)
    display_best_results(geolife_lowshot_results, "BEST GEOLIFE LOW-SHOT FINETUNE TRANSFORMER RESULTS", args.top_n)

    if geolife_lowshot_results:
        print("\nLow-shot split summary:")
        for result in geolife_lowshot_results:
            counts = extract_split_counts(Path(result["log_path"]))
            if counts:
                train_count, val_count, test_count = counts
                acc = result["test_accuracy"] * 100 if result["test_accuracy"] <= 1 else result["test_accuracy"]
                print(
                    f"  {result['experiment']}: train={train_count}, val={val_count}, test={test_count}, "
                    f"accuracy={acc:.2f}%"
                )

    # Visualisations ---------------------------------------------------------
    outputs: List[Path] = []

    training_data = {}
    if best_geolife_transformer and matching_geolife_lstm:
        training_data["geolife"] = {"transformer": best_geolife_transformer, "lstm": matching_geolife_lstm}
    if best_mobis_transformer and matching_mobis_lstm:
        training_data["mobis"] = {"transformer": best_mobis_transformer, "lstm": matching_mobis_lstm}

    if training_data:
        path = plot_training_comparison(training_data, output_dir / "original_datasets_val_comparison", show=show)
        if path:
            outputs.append(path)

    model_hist = plot_model_dataset_f1_histograms(
        (best_geolife_transformer, matching_geolife_lstm),
        (best_mobis_transformer, matching_mobis_lstm),
        output_dir / "geolife_mobis_model_f1",
        show=show,
    )
    if model_hist:
        outputs.append(model_hist)

    per_class_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    if best_finetune_transformer and matching_finetune_lstm:
        per_class_metrics["finetune_sweeps"] = {
            "transformer": extract_per_class_metrics_from_log(Path(best_finetune_transformer["log_path"])),
            "lstm": extract_per_class_metrics_from_log(Path(matching_finetune_lstm["log_path"])),
            "transformer_exp": best_finetune_transformer["experiment"],
            "lstm_exp": matching_finetune_lstm["experiment"],
        }
    if best_miniprogram_transformer and matching_miniprogram_lstm:
        per_class_metrics["miniprogram_finetune"] = {
            "transformer": extract_per_class_metrics_from_log(Path(best_miniprogram_transformer["log_path"])),
            "lstm": extract_per_class_metrics_from_log(Path(matching_miniprogram_lstm["log_path"])),
            "transformer_exp": best_miniprogram_transformer["experiment"],
            "lstm_exp": matching_miniprogram_lstm["experiment"],
        }

    if per_class_metrics:
        outputs.extend(plot_per_class_metrics(per_class_metrics, output_dir, show=show))

    window_plot = plot_window_sweep(geolife_window_results, output_dir / "geolife_window_size_accuracy", show=show)
    if window_plot:
        outputs.append(window_plot)

    lowshot_plot = plot_lowshot_results(geolife_lowshot_results, output_dir / "geolife_lowshot_accuracy", show=show)
    if lowshot_plot:
        outputs.append(lowshot_plot)

    # Summary table ---------------------------------------------------------
    summary_df = create_experiment_summary(
        geolife_transformer_results,
        matching_geolife_lstm,
        mobis_transformer_results,
        matching_mobis_lstm,
        geolife_window_results,
        finetune_transformer_results,
        matching_finetune_lstm,
        miniprogram_transformer_results,
        matching_miniprogram_lstm,
        geolife_lowshot_results,
    )

    print("=" * 80)
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print("=" * 80)
    if summary_df.empty:
        print("No results available to summarise.")
    else:
        print(summary_df.to_string(index=False))

    summary_path = output_dir / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    if outputs:
        print("Generated files:")
        for path in outputs:
            print(f"- {path.relative_to(output_dir)}")
    else:
        print("No figures were generated – check experiment availability.")


if __name__ == "__main__":
    main()
