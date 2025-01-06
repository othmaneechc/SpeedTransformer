import pandas as pd
import numpy as np
import argparse
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute descriptive statistics for 'speed' per 'label' using multiprocessing and tqdm."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file (e.g., carbonClever.csv)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the descriptive statistics CSV. Defaults to 'speed_statistics.csv' if not specified."
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs='+',
        default=[10, 25, 50, 75, 90],
        help="List of percentiles to calculate (e.g., 10 25 50 75 90). Defaults to [10, 25, 50, 75, 90]."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes for multiprocessing. Defaults to the number of CPU cores."
    )
    return parser.parse_args()

def compute_statistics_per_label(args):
    """
    Computes descriptive statistics for the 'speed' column and counts unique trips for a single label.

    Args:
        args (tuple): A tuple containing (label, speed_series, trip_count, percentiles).

    Returns:
        dict: A dictionary containing the statistics and trip count for the label.
    """
    label, speed_series, trip_count, percentiles = args

    stats = {
        'label': label,
        'count': speed_series.count(),
        'mean': speed_series.mean(),
        'std': speed_series.std(),
        'min': speed_series.min(),
        'max': speed_series.max(),
        'num_trips': trip_count  # Number of unique trips for this label
    }

    # Calculate specified percentiles
    percentile_values = speed_series.quantile([p / 100 for p in percentiles]).to_dict()
    for p, value in percentile_values.items():
        stats[f'percentile_{int(p * 100)}'] = value

    return stats

def save_statistics(stats_df, output_file):
    """
    Saves the statistics DataFrame to a CSV file.

    Args:
        stats_df (pd.DataFrame): The DataFrame containing statistics.
        output_file (str): Path to the output CSV file.
    """
    try:
        stats_df.to_csv(output_file, index=False)
        print(f"\nDescriptive statistics saved to '{output_file}'.")
    except Exception as e:
        print(f"Error saving statistics to '{output_file}': {e}")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    input_file = args.input_file

    # Determine output file path
    if args.output_file:
        output_file = args.output_file
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_speed_statistics{ext}" if ext else f"{base}_speed_statistics.csv"

    # Validate input file existence
    if not os.path.isfile(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        return

    # Load the data
    print(f"Loading data from '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{input_file}' is empty.")
        return
    except Exception as e:
        print(f"An error occurred while reading '{input_file}': {e}")
        return

    # Validate required columns
    required_columns = {'speed', 'label', 'traj_id'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Error: Input file is missing required columns: {missing_columns}")
        return

    # Drop rows with missing 'speed' or 'label' values
    initial_count = len(df)
    df = df.dropna(subset=['speed', 'label', 'traj_id'])
    final_count = len(df)
    if final_count < initial_count:
        print(f"Dropped {initial_count - final_count} rows due to missing 'speed', 'label', or 'traj_id' values.")

    # Ensure 'speed' is numeric
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['speed'])
    df['speed'] = df['speed'].astype(float)

    # Get unique labels
    labels = df['label'].unique()
    print(f"Found {len(labels)} unique labels: {labels}")

    # Prepare arguments for multiprocessing
    percentiles = args.percentiles
    pool_args = []
    for label in labels:
        speed_series = df[df['label'] == label]['speed']
        trip_count = df[df['label'] == label]['traj_id'].nunique()
        pool_args.append((label, speed_series, trip_count, percentiles))

    # Determine number of worker processes
    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"Using {num_workers} worker processes for multiprocessing...")

    # Initialize multiprocessing Pool
    with Pool(processes=num_workers) as pool:
        # Use tqdm to display a progress bar
        print("Computing descriptive statistics...")
        results = list(tqdm(pool.imap(compute_statistics_per_label, pool_args), total=len(pool_args), desc="Processing Labels"))

    # Convert results to DataFrame
    stats_df = pd.DataFrame(results)

    # Order columns
    ordered_columns = ['label', 'count', 'mean', 'std', 'min', 'max', 'num_trips']
    percentile_columns = [f'percentile_{int(p)}' for p in percentiles]
    final_columns = ordered_columns + percentile_columns
    stats_df = stats_df[final_columns]

    # Display the statistics
    print("\nDescriptive Statistics:")
    print(stats_df)

    # Save statistics to CSV
    save_statistics(stats_df, output_file)

if __name__ == "__main__":
    main()

# run the following command
# python describe_speed.py carbonClever.csv --output_file speed_statistics.csv --percentiles 10 25 50 75 90 

