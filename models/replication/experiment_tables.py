#!/usr/bin/env python3
"""Generate Markdown tables summarizing all experiments and write them to a text file."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Allow imports from the replication module
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from metrics_gen import extract_test_accuracy_from_log  # type: ignore

LOG_SPLIT_PATTERN = re.compile(r"Train:\s*(\d+),\s*Val:\s*(\d+),\s*Test:\s*(\d+)")


@dataclass
class GroupConfig:
    model_name: str
    base_path: Path
    log_candidates: Tuple[str, ...]
    default_learning_rate: Optional[str]
    default_batch_size: Optional[str]
    default_dropout: Optional[str]
    default_weight_decay: Optional[str]
    default_epochs: str
    default_early_stopping: str
    default_frozen: str


def parse_dataset_details(log_path: Path) -> Tuple[Optional[Tuple[int, int, int]], Optional[str]]:
    try:
        content = log_path.read_text()
    except OSError:
        return None, None

    matches = LOG_SPLIT_PATTERN.findall(content)
    if not matches:
        return None, None

    primary = tuple(int(value) for value in matches[0])
    note = None
    if len(matches) > 1:
        windows = tuple(int(value) for value in matches[1])
        note = "Windows: Train {0:,}, Val {1:,}, Test {2:,}".format(*windows)
    return primary, note


def parse_from_name(name: str) -> Dict[str, str]:
    params: Dict[str, str] = {}

    lr_match = re.search(r"lr([0-9.eE+-]+)", name)
    if lr_match:
        params["learning_rate"] = lr_match.group(1)

    bs_match = re.search(r"bs(\d+)", name)
    if bs_match:
        params["batch_size"] = bs_match.group(1)

    do_match = re.search(r"do([0-9.]+)", name)
    if do_match:
        params["dropout"] = do_match.group(1)

    patience_match = re.search(r"pat(\d+)", name)
    if patience_match:
        params["early_stopping"] = f"Patience {patience_match.group(1)}"

    return params


def parse_frozen_layers(name: str, default_value: str) -> str:
    mapping = {
        "freeze_attention": "Attention frozen",
        "freeze_embeddings": "Embeddings frozen",
        "freeze_feedforward": "Feedforward frozen",
        "freeze_first_half": "First half frozen",
        "freeze_last": "Last block frozen",
        "grad_unfreeze": "Gradual unfreeze",
        "reinit_last": "Reinit last block",
    }
    for key, label in mapping.items():
        if key in name:
            return label
    if "none" in name or "full" in name:
        return "None"
    return default_value


def find_log_path(exp_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    for candidate in candidates:
        candidate_path = exp_dir / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def format_dataset_cell(counts: Optional[Tuple[int, int, int]], note: Optional[str]) -> str:
    if not counts:
        return "—"
    train, val, test = counts
    cell = f"Train: {train:,}<br>Val: {val:,}<br>Test: {test:,}"
    if note:
        cell += f"<br>{note}"
    return cell


def format_accuracy(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def gather_rows(group: GroupConfig) -> List[Dict[str, str]]:
    if not group.base_path.exists():
        return []

    rows: List[Dict[str, str]] = []
    for exp_dir in sorted(group.base_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        log_path = find_log_path(exp_dir, group.log_candidates)
        if not log_path:
            continue

        accuracy = extract_test_accuracy_from_log(log_path)
        counts, note = parse_dataset_details(log_path)
        parsed = parse_from_name(exp_dir.name)

        learning_rate = parsed.get("learning_rate", group.default_learning_rate or "—")
        batch_size = parsed.get("batch_size", group.default_batch_size or "—")
        dropout = parsed.get("dropout", group.default_dropout or "—")
        weight_decay = group.default_weight_decay or "—"
        epochs = group.default_epochs
        early_stopping = parsed.get("early_stopping", group.default_early_stopping)
        frozen_layers = parse_frozen_layers(exp_dir.name, group.default_frozen)

        rows.append({
            "model": group.model_name,
            "experiment": exp_dir.name,
            "dataset": format_dataset_cell(counts, note),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "early_stopping": early_stopping,
            "frozen_layers": frozen_layers,
            "accuracy": format_accuracy(accuracy),
        })

    return rows


def render_table(title: str, rows: List[Dict[str, str]]) -> str:
    if not rows:
        return f"## {title}\n\nNo experiments found.\n\n"

    header = (
        "| Model | Experiment | Dataset size | Learning rate | Batch size | Dropout | Weight decay | "
        "Epochs (max) | Early stopping | Frozen layers | Test accuracy |"
    )
    divider = "|-------|-----------|--------------|---------------|-----------|---------|--------------|-------------|----------------|---------------|---------------|"
    lines = [f"## {title}", "", header, divider]

    for row in rows:
        line = (
            f"| {row['model']} | {row['experiment']} | {row['dataset']} | {row['learning_rate']} | "
            f"{row['batch_size']} | {row['dropout']} | {row['weight_decay']} | {row['epochs']} | "
            f"{row['early_stopping']} | {row['frozen_layers']} | {row['accuracy']} |"
        )
        lines.append(line)

    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = CURRENT_DIR.parent.parent
    models_root = repo_root / "models"

    categories = [
        (
            "Training on MOBIS",
            [
                GroupConfig(
                    model_name="SpeedTransformer",
                    base_path=models_root / "transformer" / "experiments" / "mobis_transformer_sweeps",
                    log_candidates=("train.log",),
                    default_learning_rate="1e-4",
                    default_batch_size="512",
                    default_dropout="0.1",
                    default_weight_decay="1e-4",
                    default_epochs="50",
                    default_early_stopping="Patience 7",
                    default_frozen="None",
                ),
                GroupConfig(
                    model_name="LSTM-Attention",
                    base_path=models_root / "lstm" / "experiments" / "mobis_lstm_sweeps",
                    log_candidates=("lstm.log", "train.log"),
                    default_learning_rate="5e-4",
                    default_batch_size="128",
                    default_dropout="0.1",
                    default_weight_decay="1e-4",
                    default_epochs="50",
                    default_early_stopping="Patience 7",
                    default_frozen="None",
                ),
            ],
        ),
        (
            "Training on Geolife",
            [
                GroupConfig(
                    model_name="SpeedTransformer",
                    base_path=models_root / "transformer" / "experiments" / "geolife_transformer_sweeps",
                    log_candidates=("train.log",),
                    default_learning_rate="2e-4",
                    default_batch_size="512",
                    default_dropout="0.1",
                    default_weight_decay="1e-4",
                    default_epochs="50",
                    default_early_stopping="Patience 7",
                    default_frozen="None",
                ),
                GroupConfig(
                    model_name="LSTM-Attention",
                    base_path=models_root / "lstm" / "experiments" / "geolife_lstm_sweeps",
                    log_candidates=("lstm.log", "train.log"),
                    default_learning_rate="5e-4",
                    default_batch_size="128",
                    default_dropout="0.1",
                    default_weight_decay="1e-4",
                    default_epochs="50",
                    default_early_stopping="Patience 7",
                    default_frozen="None",
                ),
            ],
        ),
        (
            "Fine-tuning MOBIS → Geolife",
            [
                GroupConfig(
                    model_name="SpeedTransformer",
                    base_path=models_root / "transformer" / "experiments" / "finetune_sweeps",
                    log_candidates=("finetune.log",),
                    default_learning_rate="1e-4",
                    default_batch_size="512",
                    default_dropout="0.2",
                    default_weight_decay="1e-4",
                    default_epochs="50",
                    default_early_stopping="Patience 7",
                    default_frozen="None",
                ),
                GroupConfig(
                    model_name="SpeedTransformer",
                    base_path=models_root / "transformer" / "experiments" / "finetune_lowshot",
                    log_candidates=("finetune.log",),
                    default_learning_rate="2e-4",
                    default_batch_size="512",
                    default_dropout="0.1",
                    default_weight_decay="1e-4",
                    default_epochs="20",
                    default_early_stopping="Patience 7",
                    default_frozen="Attention frozen",
                ),
                GroupConfig(
                    model_name="LSTM-Attention",
                    base_path=models_root / "lstm" / "experiments" / "finetune_sweeps",
                    log_candidates=("finetune.log",),
                    default_learning_rate="5e-4",
                    default_batch_size="128",
                    default_dropout="0.3",
                    default_weight_decay="1e-4",
                    default_epochs="60",
                    default_early_stopping="Patience 7",
                    default_frozen="None",
                ),
                GroupConfig(
                    model_name="LSTM-Attention",
                    base_path=models_root / "lstm" / "experiments" / "finetune_lowshot",
                    log_candidates=("finetune.log",),
                    default_learning_rate="5e-5",
                    default_batch_size="128",
                    default_dropout="0.3",
                    default_weight_decay="5e-3",
                    default_epochs="20",
                    default_early_stopping="Patience 3",
                    default_frozen="None",
                ),
            ],
        ),
        (
            "Fine-tuning MOBIS → Mini-Program",
            [
                GroupConfig(
                    model_name="SpeedTransformer",
                    base_path=models_root / "transformer" / "experiments" / "miniprogram_finetune",
                    log_candidates=("finetune.log",),
                    default_learning_rate="5e-4",
                    default_batch_size="512",
                    default_dropout="0.2",
                    default_weight_decay="1e-4",
                    default_epochs="50",
                    default_early_stopping="Patience 10",
                    default_frozen="None",
                ),
                GroupConfig(
                    model_name="LSTM-Attention",
                    base_path=models_root / "lstm" / "experiments" / "miniprogram_finetune",
                    log_candidates=("finetune.log",),
                    default_learning_rate="5e-4",
                    default_batch_size="128",
                    default_dropout="0.3",
                    default_weight_decay="1e-4",
                    default_epochs="50",
                    default_early_stopping="Patience 10",
                    default_frozen="None",
                ),
            ],
        ),
    ]

    output_lines: List[str] = []

    for title, groups in categories:
        category_rows: List[Dict[str, str]] = []
        for group in groups:
            category_rows.extend(gather_rows(group))
        category_rows.sort(key=lambda row: (row["model"], row["experiment"]))
        output_lines.append(render_table(title, category_rows))

    output_text = "\n".join(output_lines)
    output_path = CURRENT_DIR / "experiment_tables.txt"
    output_path.write_text(output_text)
    print(f"Written tables to {output_path}")


if __name__ == "__main__":
    main()
