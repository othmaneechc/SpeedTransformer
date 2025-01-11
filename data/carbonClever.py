import pandas as pd
import numpy as np
import argparse
import os
from math import radians, sin, cos, asin, sqrt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def haversine_vectorized(lon1, lat1, lon2, lat2):
    """
    Vectorized Haversine function to calculate the great-circle distance between two points 
    on the Earth specified by longitude and latitude arrays.

    Parameters:
        lon1, lat1: Arrays of longitude and latitude for point 1 in decimal degrees
        lon2, lat2: Arrays of longitude and latitude for point 2 in decimal degrees

    Returns:
        Numpy array of distances in meters between the two points.
    """
    # Convert decimal degrees to radians
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius = 6371000  # Radius of Earth in meters
    distance = earth_radius * c
    return distance

def process_single_trip(args):
    """
    Processes a single trip DataFrame:
      1) Sort by timestamp
      2) Filter by desired labels
      3) Shift coords to get prev coords/timestamps
      4) Compute distance/time diffs and speed
      5) Filter by speed thresholds
      6) Remove trip if < 3 rows

    Args:
        args: Tuple containing (group_id, group_df, desired_labels, speed_thresholds)

    Returns:
        Processed DataFrame for the trip or an empty DataFrame if it doesn't meet criteria.
    """
    group_id, trip_df, desired_labels, speed_thresholds = args

    # Sort by timestamp
    trip_df = trip_df.sort_values('timestamp').reset_index(drop=True)

    # Filter by desired labels
    trip_df = trip_df[trip_df['translated_transport'].isin(desired_labels)].reset_index(drop=True)
    if trip_df.empty:
        return pd.DataFrame([])

    # Shift to get previous longitude, latitude, and time
    trip_df['Prev_Longitude'] = trip_df['longitude'].shift(1)
    trip_df['Prev_Latitude'] = trip_df['latitude'].shift(1)
    trip_df['Prev_Time'] = trip_df['timestamp'].shift(1)

    # Ensure 'Prev_Time' is datetime and handle missing values
    trip_df['Prev_Time'] = pd.to_datetime(trip_df['Prev_Time'], errors='coerce')

    # Calculate distance using the Haversine function
    trip_df['Distance_m'] = haversine_vectorized(
        trip_df['Prev_Longitude'].values,
        trip_df['Prev_Latitude'].values,
        trip_df['longitude'].values,
        trip_df['latitude'].values
    )
    trip_df['Distance_m'] = trip_df['Distance_m'].fillna(0)

    # Calculate time difference in seconds
    trip_df['Time_diff'] = (trip_df['timestamp'] - trip_df['Prev_Time']).dt.total_seconds()
    trip_df['Time_diff'] = trip_df['Time_diff'].fillna(0)

    # Calculate speed in km/h
    trip_df['Speed_km_per_h'] = np.where(
        trip_df['Time_diff'] > 0,
        (trip_df['Distance_m'] / trip_df['Time_diff']) * 3.6,
        0
    )

    # Apply speed thresholds
    def filter_speed(row):
        label = row['translated_transport']
        speed = row['Speed_km_per_h']
        if label in speed_thresholds:
            min_speed = speed_thresholds[label]['min']
            max_speed = speed_thresholds[label]['max']
            return min_speed <= speed <= max_speed
        return False

    trip_df = trip_df[trip_df.apply(filter_speed, axis=1)].reset_index(drop=True)

    # Remove trips with fewer than 3 data points
    if len(trip_df) < 3:
        return pd.DataFrame([])

    return trip_df

def process_wrapper(args):
    """
    A top-level function (so it's pickleable) that calls `process_single_trip`.
    `args` is a tuple of (group_id, group_df, desired_labels, speed_thresholds).

    Args:
        args: Tuple containing (group_id, group_df, desired_labels, speed_thresholds)

    Returns:
        Processed DataFrame or empty DataFrame.
    """
    return process_single_trip(args)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process trajectory data to calculate speed.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing trajectory data.")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Path to the output CSV file. If not specified, '_processed' will be appended to the input file name.")
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of worker processes (defaults to CPU count).")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_file = args.input_file

    # Determine output file path
    if args.output_file:
        output_file = args.output_file
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_processed{ext}"

    # Desired labels and speed thresholds
    desired_labels = ['car', 'train', 'bike', 'walk', 'bus']
    speed_thresholds = {
        'bike':  {'min': 0.5,  'max': 80},
        'bus':   {'min': 1,    'max': 120},
        'car':   {'min': 3,    'max': 180},
        'train': {'min': 3,    'max': 350},
        'walk':  {'min': 0.1,  'max': 15},
    }

    print("Loading data...")
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

    required_columns = {'_id', 'timestamp', 'translated_transport', 'longitude', 'latitude'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Error: Input file is missing required columns: {missing_columns}")
        return

    # Drop the original 'distance' column if it exists
    if 'distance' in df.columns:
        print("Dropping the original 'distance' column from input data.")
        df.drop(columns=['distance'], inplace=True)

    # Convert 'timestamp' to datetime
    print("Parsing 'timestamp' column to datetime...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Check for any rows where 'timestamp' could not be parsed
    invalid_timestamps = df['timestamp'].isna().sum()
    if invalid_timestamps > 0:
        print(f"Warning: {invalid_timestamps} rows have invalid 'timestamp' values and will be dropped.")
        df = df.dropna(subset=['timestamp'])

    # Group by trip ID
    grouped = df.groupby('_id')
    print(grouped.head())
    # Prepare arguments for each group
    groups = []
    for group_id, group_df in grouped:
        groups.append((group_id, group_df, desired_labels, speed_thresholds))

    # Determine number of workers
    n_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print(f"Using {n_workers} worker processes...")

    # Initialize multiprocessing Pool
    with Pool(processes=n_workers) as pool:
        # Use tqdm to display a progress bar
        results = []
        try:
            for res in tqdm(pool.imap(process_wrapper, groups), total=len(groups), desc="Processing trips"):
                if not res.empty:
                    results.append(res)
        except KeyboardInterrupt:
            print("Processing interrupted by user.")
            pool.terminate()
            pool.join()
            return
        except Exception as e:
            print(f"An error occurred during multiprocessing: {e}")
            pool.terminate()
            pool.join()
            return

    if not results:
        print("No data to save after processing. Exiting.")
        return

    # Concatenate all processed trips
    processed_df = pd.concat(results, ignore_index=True)

    # Rename columns to final desired output
    processed_df.rename(columns={
        '_id': 'traj_id',
        'translated_transport': 'label',
        'Time_diff': 'time_diff',
        'Speed_km_per_h': 'speed',
        'Distance_m': 'geo_distance_m'  # Renamed for clarity
    }, inplace=True)

    # Select only the specified columns in the final DataFrame
    final_columns = ['time_diff', 'speed', 'geo_distance_m', 'traj_id', 'label']
    # Ensure all required columns are present
    for col in final_columns:
        if col not in processed_df.columns:
            final_df[col] = np.nan  # or appropriate default value

    final_df = processed_df[final_columns]

    # Optionally, rename 'geo_distance_m' back to 'distance' if preferred
    final_df.rename(columns={'geo_distance_m': 'distance'}, inplace=True)

    print(f"Saving processed data to {output_file}...")
    try:
        final_df.to_csv(output_file, index=False)
        print("Processing complete.")
    except Exception as e:
        print(f"An error occurred while saving to '{output_file}': {e}")

if __name__ == "__main__":
    main()