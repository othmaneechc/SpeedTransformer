#!/usr/bin/env python3
"""
Standalone rule-based baseline with auto-calibration.

- Reads RAW CSV with columns: traj_id, label, speed  (rename via flags if needed)
- Splits by traj_id (train/val/test) to avoid leakage.
- Builds sliding windows directly from RAW speed (optionally converts kph->m/s).
- Auto-calibrates thresholds from the TRAIN split (robust percentiles).
- Evaluates on TEST split and saves a report + confusion matrix (300 dpi).
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

DEFAULT_RULES_MPS = {
    "walk_p95_max": 1.75,
    "bike_p95_max": 2.08,
    "road_p95_min": 2.08,
    "rail_p95_min": 13.0,
    "stop_thresh": 0.3,
    "bus_stop_ratio_min": 0.20,
    "accel_std_split": 0.6
}

LABEL_SYNONYMS = {"rail": ["train"], "train": ["rail"]}


def resolve_label(name: str, labels: list[str]) -> str | None:
    for cand in [name] + LABEL_SYNONYMS.get(name, []):
        if cand in labels:
            return cand
    return None


def apply_rule_sanity(rules: dict) -> dict:
    r = dict(rules)
    r["walk_p95_max"] = max(0.3, min(r.get("walk_p95_max", 2.0), 3.0))
    r["bike_p95_max"] = max(r["walk_p95_max"] + 0.2, min(r.get("bike_p95_max", 6.0), 8.0))
    r["road_p95_min"] = max(r["bike_p95_max"], min(r.get("road_p95_min", 9.0), 18.0))
    r["rail_p95_min"] = max(r["road_p95_min"] + 1.5, min(r.get("rail_p95_min", 22.0), 40.0))
    r["stop_thresh"] = max(0.05, min(r.get("stop_thresh", 0.3), 1.0))
    r["bus_stop_ratio_min"] = max(0.05, min(r.get("bus_stop_ratio_min", 0.2), 0.7))
    r["accel_std_split"] = max(0.05, min(r.get("accel_std_split", 0.6), 3.0))
    r["bike_p95_max"] = max(r["bike_p95_max"], r["walk_p95_max"] + 0.2)
    r["road_p95_min"] = max(r["road_p95_min"], r["bike_p95_max"])
    r["rail_p95_min"] = max(r["rail_p95_min"], r["road_p95_min"] + 1.5)
    return r


def pct(a, q, default=0.0):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if a.size else default


def window_stats(v: np.ndarray, stop_thresh: float = 0.3):
    v = np.nan_to_num(v, nan=0.0)
    return {
        "p95_v": pct(v, 95, 0.0),
        "stop_ratio": float(np.mean(v < stop_thresh)) if len(v) else 1.0,
        "accel_std": float(np.std(np.diff(v))) if len(v) >= 2 else 0.0
    }


def build_class_prototypes(X, M, y_idx, class_names, speed_unit, stop_thresh):
    feats = windows_to_features(X, M, speed_unit, stop_thresh)
    y_arr = np.asarray(y_idx)
    protos = {}
    for i, label in enumerate(class_names):
        f = feats[y_arr == i]
        if f.size == 0:
            continue
        center = np.median(f, axis=0)
        iqr = np.percentile(f, 75, axis=0) - np.percentile(f, 25, axis=0)
        scale = np.maximum(iqr, np.array([0.2, 0.05, 0.05]))
        protos[label] = {"center": center, "scale": scale}
    return protos


def predict_mode_from_stats(st, labels, rules, fallback_label, prototypes=None):
    p95, stop_ratio, accel_std = st["p95_v"], st["stop_ratio"], st["accel_std"]

    def avail(name):
        r = resolve_label(name, labels)
        return r if r else fallback_label

    def nearest():
        if not prototypes:
            return fallback_label
        vec = np.array([p95, stop_ratio, accel_std])
        best = float("inf"); best_name = fallback_label
        for name, proto in prototypes.items():
            c, s = proto["center"], proto["scale"]
            score = np.sum(np.abs(vec - c) / np.maximum(s, 1e-3))
            if score < best:
                best, best_name = score, name
        return best_name

    # Main decision
    if p95 >= rules["rail_p95_min"]:
        return avail("rail")
    if p95 <= rules["walk_p95_max"]:
        return avail("walk")
    if p95 <= rules["bike_p95_max"]:
        return avail("bike")
    if p95 >= rules["road_p95_min"]:
        if stop_ratio >= rules["bus_stop_ratio_min"] and accel_std <= rules["accel_std_split"] * 1.2:
            return avail("bus")
        return avail("car")

    # Fallback — choose nearest prototype if uncertain
    return nearest()


def windows_to_features(X, M, speed_unit, stop_thresh):
    feats = []
    to_mps = speed_unit.lower() == "kph"
    for win, mask in zip(X, M):
        v = np.asarray(win, float)
        if to_mps:
            v *= 1 / 3.6
        if mask is not None:
            v = v[~mask]
        s = window_stats(v, stop_thresh)
        feats.append([s["p95_v"], s["stop_ratio"], s["accel_std"]])
    return np.asarray(feats, float)


def split_traj_ids(all_ids, test_size, val_size, seed):
    train, temp = train_test_split(all_ids, test_size=(val_size + test_size), random_state=seed)
    val_ratio = val_size / (val_size + test_size)
    val, test = train_test_split(temp, test_size=(1 - val_ratio), random_state=seed)
    return set(train), set(val), set(test)


def iter_grouped_windows(speed_vec, label, win_size, stride):
    n = len(speed_vec); i = 0
    while i < n:
        j = i + win_size
        win = speed_vec[i:j]
        if len(win) < win_size:
            pad = np.zeros(win_size); pad[:len(win)] = win
            mask = np.zeros(win_size, bool); mask[len(win):] = True
            yield pad, mask, label; break
        else:
            mask = np.zeros(win_size, bool)
            yield win, mask, label
            if j >= n: break
        i += stride


def build_windows_from_csv(path, traj_col, label_col, win_size, stride, chunksize, keep_ids, speed_col="speed"):
    X, M, y = [], [], []
    usecols = [traj_col, label_col, speed_col]
    for chunk in tqdm(pd.read_csv(path, usecols=usecols, chunksize=chunksize), desc="Building windows"):
        c = chunk[chunk[traj_col].isin(keep_ids)]
        for _, g in c.groupby(traj_col):
            speed = g[speed_col].to_numpy(float)
            label = g[label_col].iloc[0]
            for win, mask, lab in iter_grouped_windows(speed, label, win_size, stride):
                X.append(win); M.append(mask); y.append(lab)
    return X, M, y


def autocalibrate_rules(Xtr, Mtr, ytr, classes, base_rules, speed_unit):
    to_mps = (speed_unit.lower() == "kph")
    kph_to_mps = 1 / 3.6
    stats_by_cls = {c: {"p95": [], "stop": [], "acc": []} for c in classes}
    for win, mask, lab in zip(Xtr, Mtr, ytr):
        v = np.asarray(win, float)
        if to_mps: v *= kph_to_mps
        if mask is not None: v = v[~mask]
        st = window_stats(v, base_rules["stop_thresh"])
        if lab in stats_by_cls:
            stats_by_cls[lab]["p95"].append(st["p95_v"])
            stats_by_cls[lab]["stop"].append(st["stop_ratio"])
            stats_by_cls[lab]["acc"].append(st["accel_std"])

    rules = dict(base_rules)
    w, b, r, bu, c = [resolve_label(x, classes) for x in ["walk", "bike", "rail", "bus", "car"]]

    if w and stats_by_cls[w]["p95"]:
        rules["walk_p95_max"] = pct(stats_by_cls[w]["p95"], 75, rules["walk_p95_max"])
    if b and stats_by_cls[b]["p95"]:
        rules["bike_p95_max"] = max(pct(stats_by_cls[b]["p95"], 75, rules["bike_p95_max"]),
                                    rules["walk_p95_max"] + 0.6)
    if r and stats_by_cls[r]["p95"]:
        rules["rail_p95_min"] = max(pct(stats_by_cls[r]["p95"], 35, rules["rail_p95_min"]),
                                    rules["bike_p95_max"] + 1.0)
    road_pool = []
    if c: road_pool += stats_by_cls[c]["p95"]
    if bu: road_pool += stats_by_cls[bu]["p95"]
    if road_pool:
        rules["road_p95_min"] = max(rules["bike_p95_max"], pct(road_pool, 20, rules["road_p95_min"]))
    if bu and stats_by_cls[bu]["stop"]:
        rules["bus_stop_ratio_min"] = max(0.05, min(0.9, pct(stats_by_cls[bu]["stop"], 50, 0.2) - 0.05))
    if bu and c and stats_by_cls[bu]["acc"] and stats_by_cls[c]["acc"]:
        medb, medc = pct(stats_by_cls[bu]["acc"], 50, 0.6), pct(stats_by_cls[c]["acc"], 50, 0.6)
        rules["accel_std_split"] = max(0.05, min(2.0, 0.6 * medb + 0.4 * medc))

    if speed_unit.lower() == "mps":
        rules["stop_thresh"] = max(rules["stop_thresh"], 0.5)

    return apply_rule_sanity(rules)


# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--traj_id_column", default="traj_id")
    ap.add_argument("--target_column", default="label")
    ap.add_argument("--speed_col", default="speed")
    ap.add_argument("--chunksize", type=int, default=10**6)
    ap.add_argument("--window_size", type=int, default=200)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--random_state", type=int, default=316)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--label_encoder_path", required=True)
    ap.add_argument("--speed_unit", default="mps", choices=["mps", "kph"])
    ap.add_argument("--out_dir", default="rule_outputs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    le = joblib.load(args.label_encoder_path)
    classes = list(le.classes_)

    # Split trajectory IDs
    ids = set()
    for chunk in tqdm(pd.read_csv(args.data_path, usecols=[args.traj_id_column], chunksize=args.chunksize),
                      desc="Collecting traj_ids"):
        ids.update(chunk[args.traj_id_column].unique())
    train_ids, _, test_ids = split_traj_ids(sorted(ids), args.test_size, args.val_size, args.random_state)

    Xtr, Mtr, ytr = build_windows_from_csv(args.data_path, args.traj_id_column,
                                           args.target_column, args.window_size,
                                           args.stride, args.chunksize, train_ids, args.speed_col)
    Xte, Mte, yte = build_windows_from_csv(args.data_path, args.traj_id_column,
                                           args.target_column, args.window_size,
                                           args.stride, args.chunksize, test_ids, args.speed_col)

    Xtr_keep, Mtr_keep, y_train = [], [], []
    for win, mask, lab in zip(Xtr, Mtr, ytr):
        try:
            y_idx = int(le.transform([lab])[0])
            Xtr_keep.append(win); Mtr_keep.append(mask); y_train.append(y_idx)
        except ValueError:
            continue

    y_true, keep = [], []
    for i, lab in enumerate(yte):
        try:
            y_true.append(int(le.transform([lab])[0])); keep.append(i)
        except ValueError:
            continue
    Xte = [Xte[i] for i in keep]; Mte = [Mte[i] for i in keep]

    fallback_label = classes[int(np.argmax(np.bincount(y_train, minlength=len(classes))))]

    rules = autocalibrate_rules(Xtr, Mtr, ytr, classes, DEFAULT_RULES_MPS, args.speed_unit)
    prototypes = build_class_prototypes(Xtr_keep, Mtr_keep, y_train, classes, args.speed_unit, rules["stop_thresh"])
    (out_dir / "rules_used.json").write_text(json.dumps(rules, indent=2))
    (out_dir / "prototypes.json").write_text(json.dumps({
        k: {"center": v["center"].tolist(), "scale": v["scale"].tolist()} for k, v in prototypes.items()
    }, indent=2))

    print("\n[Rules Used]\n" + "\n".join([f"{k}: {v:.3f}" for k, v in rules.items()]) + "\n")

    # Prediction
    to_mps = (args.speed_unit.lower() == "kph")
    kph_to_mps = 1 / 3.6
    y_pred = []
    for win, mask in zip(Xte, Mte):
        v = np.asarray(win, float)
        if to_mps: v *= kph_to_mps
        if mask is not None: v = v[~mask]
        st = window_stats(v, rules["stop_thresh"])
        name = predict_mode_from_stats(st, classes, rules, fallback_label, prototypes)
        y_pred.append(int(np.where(np.array(classes) == name)[0][0]))

    acc = accuracy_score(y_true, y_pred)
    print(f"[Rule Baseline] Test Accuracy: {acc:.4f}")
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=1)
    print(report)
    (out_dir / "report.txt").write_text(f"Accuracy: {acc:.4f}\n\n{report}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Rule Baseline — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_rule.png", dpi=300)
    plt.close()
    print(f"[OK] Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
