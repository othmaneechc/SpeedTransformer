#!/usr/bin/env python3
"""
Standalone rule-based baseline with optional auto-calibration.

- No dependency on your training code.
- Reads RAW CSV with columns: traj_id, label, speed  (rename via flags if needed)
- Splits by traj_id (train/val/test) to avoid leakage.
- Builds sliding windows directly from RAW speed (optionally converts kph->m/s).
- Optional: auto-calibrate thresholds from the TRAIN split (robust percentiles).
- Evaluates on TEST split and saves a report + confusion matrix (300 dpi).

Examples:
  # Geolife (m/s)
    python rule_baseline.py \
    --data_path "/data/A-SpeedTransformer/data/geolife_processed.csv" \
    --label_encoder_path "/data/A-SpeedTransformer/models/rule_based/label_encoder.joblib" \
    --out_dir "/data/A-SpeedTransformer/models/rule_based/experiments/rule_geolife" \
    --auto_calibrate

  # MOBIS (m/s)
    python rule_baseline.py \
    --data_path "/data/A-SpeedTransformer/data/mobis_processed.csv" \
    --label_encoder_path "/data/A-SpeedTransformer/models/rule_based/label_encoder.joblib" \
    --out_dir "/data/A-SpeedTransformer/models/rule_based/experiments/rule_mobis" \
    --auto_calibrate
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------ Defaults & helpers ------------------

DEFAULT_CLASSES = ["walk", "bike", "bus", "car", "rail"]

# Thresholds in m/s (1 m/s = 3.6 km/h)
# DEFAULT_RULES_MPS = {
#     "walk_p95_max":     2.0,   # ~7.2 km/h
#     "bike_p95_max":     6.0,   # ~21.6 km/h
#     "road_p95_min":     6.0,   # >= this => motorized road (bus/car)
#     "rail_p95_min":    15.0,   # ~54 km/h
#     "stop_thresh":      0.3,   # < this = stopped
#     "bus_stop_ratio_min": 0.20,
#     "accel_std_split":   0.6,  # lower -> bus, higher -> car
# }

# 1 m/s = 3.6 km/h
DEFAULT_RULES_MPS = {
    "walk_p95_max": 1.75,          # ≤ 6.3 km/h → walk
    "bike_p95_max": 2.08,          # ≤ 7.5 km/h → bike
    "road_p95_min": 2.08,          # ≥ 7.5 km/h → motorized
    "rail_p95_min": 41.7,          # ≥ 150 km/h → train
    "stop_thresh": 0.3,            # < 0.3 m/s considered stopped
    "bus_stop_ratio_min": 0.20,    # ≥ 20 % stops → bus
    "accel_std_split": 0.6         # smoother accel → bus, jerkier → car
}

def pct(a, q, default=0.0):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return default
    return float(np.percentile(a, q))

def window_stats(raw_speed_1d: np.ndarray, stop_thresh: float = 0.3):
    v = np.asarray(raw_speed_1d, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    p95_v = pct(v, 95, default=0.0)
    stop_ratio = float(np.mean(v < stop_thresh)) if v.size else 1.0
    accel_std = float(np.std(np.diff(v))) if v.size >= 2 else 0.0
    return {"p95_v": p95_v, "stop_ratio": stop_ratio, "accel_std": accel_std}

def predict_mode_from_stats(stats, labels, rules):
    """Return nearest available label name from labels[] using calibrated rules."""
    def avail(name): return name if name in labels else labels[0]
    p95 = stats["p95_v"]; stop_ratio = stats["stop_ratio"]; accel_std = stats["accel_std"]

    if p95 <= rules["walk_p95_max"]:
        return avail("walk")
    if p95 <= rules["bike_p95_max"]:
        return avail("bike")
    if p95 >= rules["rail_p95_min"]:
        return avail("rail")
    if p95 >= rules["road_p95_min"]:
        if stop_ratio >= rules["bus_stop_ratio_min"] and accel_std < rules["accel_std_split"]:
            return avail("bus")
        return avail("car")
    return avail("car")

# ------------------ Windowing & split ------------------

def split_traj_ids(all_ids, test_size, val_size, random_state):
    train_ids, temp_ids = train_test_split(
        all_ids, test_size=(val_size + test_size), random_state=random_state, shuffle=True
    )
    val_ratio_adj = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.5
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=(1 - val_ratio_adj), random_state=random_state, shuffle=True
    )
    return set(train_ids), set(val_ids), set(test_ids)

def iter_grouped_windows(speed_vec, label, window_size, stride):
    n = len(speed_vec)
    i = 0
    while i < n:
        j = i + window_size
        win = speed_vec[i:j]
        if len(win) < window_size:
            pad = np.zeros(window_size, dtype=float)
            pad[:len(win)] = win
            mask = np.zeros(window_size, dtype=bool)
            mask[len(win):] = True
            yield pad, mask, label
            break
        else:
            mask = np.zeros(window_size, dtype=bool)
            yield win, mask, label
            if j >= n: break
        i += stride

def build_windows_from_csv(
    data_path: str,
    traj_id_col: str,
    label_col: str,
    window_size: int,
    stride: int,
    chunksize: int,
    keep_ids: set,
    speed_col: str = "speed",
):
    X, M, y = [], [], []
    usecols = [traj_id_col, label_col, speed_col]
    for chunk in tqdm(pd.read_csv(data_path, usecols=usecols, chunksize=chunksize), desc="Building windows"):
        c = chunk[chunk[traj_id_col].isin(keep_ids)].copy()
        if c.empty:
            continue
        for _, g in c.groupby(traj_id_col, sort=False):
            speed = g[speed_col].to_numpy(dtype=float)
            label = g[label_col].iloc[0]
            for win, mask, lab in iter_grouped_windows(speed, label, window_size, stride):
                X.append(win); M.append(mask); y.append(lab)
    return X, M, y

# ------------------ Auto-calibration ------------------

def autocalibrate_rules(
    Xtr: list[np.ndarray],
    Mtr: list[np.ndarray],
    ytr: list[str],
    class_names: list[str],
    base_rules: dict,
    speed_unit: str,
    report_path: Path | None = None,
) -> dict:
    """
    Learn dataset-aware thresholds from TRAIN windows using robust percentiles:
      - walk_p95_max: 75th percentile of p95 among walk
      - bike_p95_max: 75th percentile of p95 among bike, but ≥ walk cutoff + margin
      - rail_p95_min: 35th percentile of p95 among rail (or 25th if many rails)
      - bus_stop_ratio_min: median(stop_ratio) among bus minus small margin
      - accel_std_split: mid-quantile between bus and car accel_std (e.g., mean of medians)
      - stop_thresh: keep base (or raise slightly if speeds are noisy and unit is m/s)
    If a class is absent, fallback to base_rules for that boundary.
    """
    to_mps = (speed_unit.lower() == "kph")
    kph_to_mps = 1.0 / 3.6

    # Collect per-class stats
    stats_by_cls = {c: {"p95": [], "stop": [], "accstd": []} for c in class_names}
    for win, mask, lab in zip(Xtr, Mtr, ytr):
        v = np.asarray(win, dtype=float)
        if to_mps:
            v *= kph_to_mps
        if mask is not None:
            v = v[~mask.astype(bool)]
        st = window_stats(v, stop_thresh=base_rules["stop_thresh"])
        bucket = stats_by_cls.get(lab)
        if bucket is not None:
            bucket["p95"].append(st["p95_v"])
            bucket["stop"].append(st["stop_ratio"])
            bucket["accstd"].append(st["accel_std"])

    # Start from base rules; adjust with observed distributions
    rules = dict(base_rules)

    # Walk cutoff from walk p95 (75th percentile)
    if "walk" in stats_by_cls and len(stats_by_cls["walk"]["p95"]) > 10:
        rules["walk_p95_max"] = pct(stats_by_cls["walk"]["p95"], 75, default=rules["walk_p95_max"])

    # Bike cutoff from bike p95 (75th), keep ≥ walk + margin
    if "bike" in stats_by_cls and len(stats_by_cls["bike"]["p95"]) > 10:
        bike75 = pct(stats_by_cls["bike"]["p95"], 75, default=rules["bike_p95_max"])
        rules["bike_p95_max"] = max(bike75, rules["walk_p95_max"] + 0.6)  # +0.6 m/s margin

    # Rail min from rail p95 (35th) — more conservative if little rail
    if "rail" in stats_by_cls and len(stats_by_cls["rail"]["p95"]) > 10:
        rail_q = 35 if len(stats_by_cls["rail"]["p95"]) > 200 else 25
        rules["rail_p95_min"] = max(
            pct(stats_by_cls["rail"]["p95"], rail_q, default=rules["rail_p95_min"]),
            rules["bike_p95_max"] + 1.0,  # ensure ordering
        )

    # Road min should sit between bike and rail; keep at least bike cutoff
    rules["road_p95_min"] = max(rules["road_p95_min"], rules["bike_p95_max"])

    # Bus stop ratio from median(bus) minus a small margin
    if "bus" in stats_by_cls and len(stats_by_cls["bus"]["stop"]) > 10:
        bus_med_stop = pct(stats_by_cls["bus"]["stop"], 50, default=rules["bus_stop_ratio_min"])
        rules["bus_stop_ratio_min"] = max(0.05, min(0.9, bus_med_stop - 0.05))

    # Accel split between bus and car: average of medians
    have_bus = "bus" in stats_by_cls and len(stats_by_cls["bus"]["accstd"]) > 10
    have_car = "car" in stats_by_cls and len(stats_by_cls["car"]["accstd"]) > 10
    if have_bus and have_car:
        med_bus = pct(stats_by_cls["bus"]["accstd"], 50, default=rules["accel_std_split"])
        med_car = pct(stats_by_cls["car"]["accstd"], 50, default=rules["accel_std_split"])
        # place split closer to bus for safety, but not above car median
        rules["accel_std_split"] = max(0.05, min(2.0, (0.6 * med_bus + 0.4 * med_car)))
    elif have_bus:
        rules["accel_std_split"] = pct(stats_by_cls["bus"]["accstd"], 60, default=rules["accel_std_split"])
    elif have_car:
        rules["accel_std_split"] = pct(stats_by_cls["car"]["accstd"], 40, default=rules["accel_std_split"])

    # If speeds are noisy (m/s), a slightly higher stop threshold helps buses
    if speed_unit.lower() == "mps":
        rules["stop_thresh"] = max(rules["stop_thresh"], 0.5)

    # Ensure ordering coherence
    rules["bike_p95_max"] = max(rules["bike_p95_max"], rules["walk_p95_max"] + 0.3)
    rules["rail_p95_min"] = max(rules["rail_p95_min"], rules["bike_p95_max"] + 0.8)
    rules["road_p95_min"] = max(rules["road_p95_min"], rules["bike_p95_max"])

    # Optionally write a calibration report
    if report_path is not None:
        rep = {
            "final_rules_mps": rules,
            "counts": {c: {k: len(v) for k, v in buckets.items()}
                       for c, buckets in stats_by_cls.items()},
            "notes": "Thresholds derived from TRAIN split using robust percentiles."
        }
        report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    return rules

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, type=str)
    ap.add_argument("--traj_id_column", default="traj_id", type=str)
    ap.add_argument("--target_column", default="label", type=str)
    ap.add_argument("--speed_col", default="speed", type=str)

    ap.add_argument("--chunksize", default=10**6, type=int)
    ap.add_argument("--window_size", default=200, type=int)
    ap.add_argument("--stride", default=25, type=int)
    ap.add_argument("--random_state", default=316, type=int)
    ap.add_argument("--test_size", default=0.15, type=float)
    ap.add_argument("--val_size", default=0.15, type=float)

    ap.add_argument("--label_encoder_path", required=True, type=str)
    ap.add_argument("--speed_unit", default="mps", choices=["mps", "kph"],
                    help="Unit of the RAW CSV speed column. Rules use m/s.")

    ap.add_argument("--auto_calibrate", action="store_true",
                    help="Derive thresholds from TRAIN split (recommended).")
    ap.add_argument("--rules_json", type=str, default=None,
                    help="Optional manual overrides as JSON (applied after auto-calibration).")
    ap.add_argument("--calibration_report", type=str, default=None,
                    help="Optional path to save a JSON report of learned thresholds.")

    ap.add_argument("--out_dir", default="rule_outputs", type=str)

    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Label encoder (for metrics)
    le = joblib.load(args.label_encoder_path)
    class_names = list(le.classes_)

    # 1) Collect unique traj_ids
    traj_ids = set()
    for chunk in tqdm(pd.read_csv(args.data_path, usecols=[args.traj_id_column], chunksize=args.chunksize),
                      desc="Collecting traj_ids"):
        traj_ids.update(chunk[args.traj_id_column].unique())
    traj_ids = sorted(traj_ids)

    # 2) Split by traj_id
    train_ids, val_ids, test_ids = split_traj_ids(traj_ids, args.test_size, args.val_size, args.random_state)

    # 3) Build TRAIN & TEST windows from RAW speed
    Xtr, Mtr, ytr = build_windows_from_csv(
        data_path=args.data_path,
        traj_id_col=args.traj_id_column,
        label_col=args.target_column,
        window_size=args.window_size,
        stride=args.stride,
        chunksize=args.chunksize,
        keep_ids=train_ids,
        speed_col=args.speed_col,
    )
    Xte, Mte, yte_labels = build_windows_from_csv(
        data_path=args.data_path,
        traj_id_col=args.traj_id_column,
        label_col=args.target_column,
        window_size=args.window_size,
        stride=args.stride,
        chunksize=args.chunksize,
        keep_ids=test_ids,
        speed_col=args.speed_col,
    )

    # 4) Convert true labels (test) -> ints (skip unseen)
    y_true = []
    keep_idx = []
    for i, lab in enumerate(yte_labels):
        try:
            y_true.append(int(le.transform([lab])[0]))
            keep_idx.append(i)
        except ValueError:
            continue
    Xte = [Xte[i] for i in keep_idx]
    Mte = [Mte[i] for i in keep_idx]

    # 5) Build rules (auto-calibrate + manual override)
    rules = dict(DEFAULT_RULES_MPS)
    if args.auto_calibrate:
        calib_report = Path(args.calibration_report) if args.calibration_report else None
        rules = autocalibrate_rules(
            Xtr=Xtr, Mtr=Mtr, ytr=ytr,
            class_names=class_names,
            base_rules=rules,
            speed_unit=args.speed_unit,
            report_path=calib_report
        )
    if args.rules_json:
        # Allow final manual tweaks on top of calibration
        rules.update(json.loads(args.rules_json))

    # Save final rules used
    (out_dir / "rules_used.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")

    # 6) Predict TEST
    to_mps = (args.speed_unit.lower() == "kph")
    kph_to_mps = 1.0 / 3.6

    y_pred = []
    for win, mask in zip(Xte, Mte):
        v = np.asarray(win, dtype=float)
        if to_mps: v *= kph_to_mps
        if mask is not None: v = v[~mask.astype(bool)]
        st = window_stats(v, stop_thresh=rules["stop_thresh"])
        pred_name = predict_mode_from_stats(st, labels=class_names, rules=rules)
        pred_idx = int(np.where(np.array(class_names) == pred_name)[0][0])
        y_pred.append(pred_idx)

    # 7) Metrics & plots
    acc = accuracy_score(y_true, y_pred)
    print(f"[Rule Baseline] Test Accuracy: {acc:.4f}")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(report)
    (out_dir / "report.txt").write_text(f"Accuracy: {acc:.4f}\n\n{report}\n", encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Rule Baseline — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_rule.png", dpi=300)
    plt.close()

    print(f"[OK] Saved outputs to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()