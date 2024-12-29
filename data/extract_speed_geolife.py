import pandas as pd
import numpy as np
from math import radians
import argparse
from tqdm import tqdm
import os

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
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius = 6371000  # Radius of Earth in meters
    distance = earth_radius * c
    return distance

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate speed from Geolife Trajectories data.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file containing Geolife Trajectories data."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the output CSV file. If not specified, '_processed' will be appended to the input file name."
    )
    return parser.parse_args()

def process_data(input_file, output_file):
    """
    Processes the Geolife Trajectories data to calculate speed, retain specified columns, 
    merge labels, and filter based on speed thresholds.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
    """
    # Load the data
    print("Loading data...")
    df = pd.read_csv(input_file)

    # Parse the 'Coordinate' column into 'Longitude' and 'Latitude'
    print("Parsing coordinates...")
    df[['Longitude', 'Latitude']] = df['Coordinate'].str.strip('()').str.split(',', expand=True).astype(float)
    df.drop('Coordinate', axis=1, inplace=True)

    # Merge labels
    print("Merging labels...")
    df['Mode'] = df['Mode'].replace({'taxi': 'car', 'subway': 'train'})

    # Retain only desired labels
    desired_labels = ['car', 'train', 'bike', 'walk', 'bus']
    df = df[df['Mode'].isin(desired_labels)]

    # Sort the DataFrame by 'Trip_id' and 'Time'
    print("Sorting data...")
    df.sort_values(by=['Trip_id', 'Time'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Shift the 'Longitude' and 'Latitude' to get previous coordinates within each 'Trip_id'
    print("Shifting coordinates and time...")
    df['Prev_Longitude'] = df.groupby('Trip_id')['Longitude'].shift(1)
    df['Prev_Latitude'] = df.groupby('Trip_id')['Latitude'].shift(1)
    df['Prev_Time'] = df.groupby('Trip_id')['Time'].shift(1)

    # Calculate distance using the vectorized Haversine function
    print("Calculating distances...")
    df['Distance_m'] = haversine_vectorized(
        df['Prev_Longitude'].values,
        df['Prev_Latitude'].values,
        df['Longitude'].values,
        df['Latitude'].values
    )
    df['Distance_m'] = df['Distance_m'].fillna(0)

    # Calculate time difference in seconds
    print("Calculating time differences...")
    df['Time_diff'] = df['Time'] - df['Prev_Time']
    df['Time_diff'] = df['Time_diff'].fillna(0)

    # Calculate speed in km/h
    print("Calculating speeds...")
    df['Speed_km_per_h'] = np.where(
        df['Time_diff'] > 0,
        (df['Distance_m'] / df['Time_diff']) * 3.6,
        0
    )

    # Apply speed thresholds for each label
    print("Applying speed thresholds...")
    speed_thresholds = {
        'bike': {'min': 0.5, 'max': 80},
        'bus': {'min': 1, 'max': 120},
        'car': {'min': 3, 'max': 180},
        'train': {'min': 3, 'max': 350},
        'walk': {'min': 0.1, 'max': 15}
    }

    def filter_speed(row):
        label = row['Mode']
        speed = row['Speed_km_per_h']
        if label in speed_thresholds:
            min_speed = speed_thresholds[label]['min']
            max_speed = speed_thresholds[label]['max']
            return min_speed <= speed <= max_speed
        return False

    df = df[df.apply(filter_speed, axis=1)]

    # ---- NEW STEP: Remove trips with fewer than 3 points ----
    print("Removing trips with fewer than 3 rows...")
    print(len(df))
    df = df.groupby('Trip_id').filter(lambda x: len(x) >= 3)
    print(len(df))

    # Rename columns as per requirement
    print("Renaming columns...")
    df.rename(columns={
        'Time_diff': 'time_diff',
        'Speed_km_per_h': 'speed',
        'Distance_m': 'distance',
        'Trip_id': 'traj_id',
        'Mode': 'label'
    }, inplace=True)

    # Select only the specified columns
    print("Selecting required columns...")
    final_df = df[['time_diff', 'speed', 'distance', 'traj_id', 'label']]

    # Save the processed data
    print(f"Saving processed data to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print("Processing complete.")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    input_file = args.input_file

    # Determine output file path
    if args.output_file:
        output_file = args.output_file
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_processed{ext}"

    # Process the data
    process_data(input_file, output_file)

if __name__ == "__main__":
    main()
