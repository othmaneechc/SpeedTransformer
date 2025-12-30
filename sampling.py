from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_time_diff(csv_path: Path) -> pd.Series:
    """Load the time_diff column as a numeric series."""
    series = pd.read_csv(csv_path, usecols=["time_diff"])["time_diff"]
    series = pd.to_numeric(series, errors="coerce", downcast="float").dropna()
    return series[series > 0]


def remove_upper_outliers(series: pd.Series, upper_quantile: float = 0.99) -> tuple[pd.Series, float, int]:
    """
    Remove upper outliers using a quantile cutoff.

    Returns the filtered series, the cutoff value, and the number of rows dropped.
    """
    cutoff = series.quantile(upper_quantile)
    filtered = series[series <= cutoff]
    dropped = len(series) - len(filtered)
    return filtered, cutoff, dropped


def plot_histograms(data: dict[str, pd.Series], output_path: Path) -> None:
    """Plot one histogram per dataset on a single figure."""
    global_min = min(series.min() for series in data.values())
    global_max = max(series.max() for series in data.values())
    if global_min == global_max:
        global_max = global_min + 1e-6  # avoid zero-width bins

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=False)

    for ax, (label, series) in zip(axes, data.items()):
        bins = 50
        ax.hist(
            series,
            bins=bins,
            range=(global_min, global_max),
            color="#4C6FFF",
            edgecolor="black",
            alpha=0.75,
        )
        ax.set_yscale("log")
        ax.set_title(f"{label}")
        ax.set_xlabel("Sampling Interval (seconds)")
        ax.set_ylabel("Count (log scale)")
        ax.set_xlim(global_min, global_max)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.suptitle("Sampling-frequency distributions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    datasets = {
        "Geolife": base_dir / "data" / "geolife_processed.csv",
        "MiniProgram": base_dir / "data" / "miniprogram_balanced.csv",
        "MOBIS": base_dir / "data" / "mobis_processed.csv",
    }

    time_diff_data = {}
    for name, path in datasets.items():
        series = load_time_diff(path)
        filtered, cutoff, dropped = remove_upper_outliers(series)
        del series
        time_diff_data[name] = filtered

        print(f"=== {name} time_diff describe (<= {cutoff:.3f}, dropped {dropped}) ===")
        print(filtered.describe())
        print()

    output_path = base_dir / "sampling_frequency_histograms.png"
    plot_histograms(time_diff_data, output_path)
    print(f"Saved histograms to {output_path}")


if __name__ == "__main__":
    main()
