import pandas as pd
import os
from tqdm import tqdm
import time
import multiprocessing
import glob
import shutil
import argparse

def process_user(args):
    """
    Processes the trajectory and labels data for a single user and writes the output to a temporary CSV file.

    Args:
        args (tuple): A tuple containing user_path, user_id, and temp_folder.

    Returns:
        str or None: Path to the temporary CSV file if processing is successful, else None.
    """
    user_path, user_id, temp_folder = args
    temp_csv_path = os.path.join(temp_folder, f"{user_id}.csv")

    trajectory_folder = os.path.join(user_path, 'Trajectory')
    labels_path = os.path.join(user_path, 'labels.txt')

    # Check if necessary files/folders exist
    if not os.path.isdir(trajectory_folder) or not os.path.isfile(labels_path):
        return None

    # Load transportation mode labels
    try:
        transport_df = pd.read_csv(labels_path, delim_whitespace=True)
    except:
        return None

    # Combine 'Start' and 'Time', 'End' and 'Time.1' into datetime
    try:
        transport_df['Start'] = pd.to_datetime(transport_df['Start'] + ' ' + transport_df['Time'])
        transport_df['End'] = pd.to_datetime(transport_df['End'] + ' ' + transport_df['Time.1'])
    except:
        return None

    # Check for 'Transportation' column
    if 'Transportation' not in transport_df.columns:
        return None

    # Drop unnecessary columns
    transport_df = transport_df.drop(columns=['Time', 'Time.1', 'Mode'], errors='ignore')

    # Initialize a list to store trajectory data
    trajectory_data = []

    # Iterate through each .plt file in the Trajectory folder
    plt_files = [f for f in os.listdir(trajectory_folder) if f.endswith('.plt') and not f.startswith('._')]
    for file_name in plt_files:
        file_path = os.path.join(trajectory_folder, file_name)

        # Read the PLT file, skipping the first 6 header lines
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()[6:]
        except:
            continue

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 7:
                trajectory_data.append(parts)

    if not trajectory_data:
        return None

    # Convert trajectory data to DataFrame
    df = pd.DataFrame(trajectory_data, columns=['Latitude', 'Longitude', '0', 'Altitude', 'Days', 'Date', 'Time'])

    df = df.drop(columns=['0', 'Altitude', 'Days'], errors='ignore')

    # Combine 'Date' and 'Time' into a single datetime column
    df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

    df = df.drop(columns=['Date'], errors='ignore')

    # Drop rows with invalid datetime
    df = df.dropna(subset=['Time'])
    if df.empty:
        return None

    # Create 'Coordinate' column as (Longitude, Latitude) tuple
    try:
        df['Coordinate'] = df.apply(lambda row: (float(row['Longitude']), float(row['Latitude'])), axis=1)
    except:
        return None

    df = df.drop(columns=['Longitude', 'Latitude'], errors='ignore')

    # Sort by Time
    df = df.sort_values(by='Time').reset_index(drop=True)

    # Initialize 'Mode' and 'Trip_id' columns
    df['Mode'] = -1
    df['Trip_id'] = -1

    # Match trajectory points with transportation modes
    for idx, row in df.iterrows():
        time_point = row['Time']
        # Find matching time intervals in the transport dataframe
        trip_match = transport_df[(transport_df['Start'] <= time_point) & (transport_df['End'] >= time_point)]
        if not trip_match.empty:
            df.at[idx, 'Mode'] = trip_match['Transportation'].values[0]
            df.at[idx, 'Trip_id'] = trip_match.index[0]

    # Filter out rows without a matching Trip_id
    df = df[df['Trip_id'] >= 0].reset_index(drop=True)
    if df.empty:
        return None

    # Calculate TimeDifference in seconds for each trip
    df['TimeDifference'] = df.groupby('Trip_id')['Time'].transform(lambda x: (x - x.iloc[0]).dt.total_seconds())
    df['TimeDifference'] = df['TimeDifference'].astype(int)

    # Replace 'Time' with 'TimeDifference'
    df = df.drop(columns=['Time'])
    df['Time'] = df['TimeDifference']
    df = df.drop(columns=['TimeDifference'])

    # Generate unique Trip_id with user identifier
    df['Trip_id'] = df.groupby('Trip_id').ngroup()
    df['Trip_id'] = df['Trip_id'].apply(lambda x: f'Geolife_{user_id}_{x:03d}')

    # Add user identifier as a separate column
    df['User'] = user_id

    # Write the processed DataFrame to a temporary CSV file
    try:
        df.to_csv(temp_csv_path, index=False)
        return temp_csv_path
    except:
        return None

def initialize_temp_folder(temp_folder):
    """
    Initializes the temporary folder by creating it if it doesn't exist
    or cleaning it if it already exists.

    Args:
        temp_folder (str): Path to the temporary folder.
    """
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

def concatenate_csv_files(temp_folder, output_file):
    """
    Concatenates all CSV files in the temporary folder into a single CSV file.

    Args:
        temp_folder (str): Path to the temporary folder containing individual CSV files.
        output_file (str): Path to the final consolidated CSV file.
    """
    temp_csv_files = glob.glob(os.path.join(temp_folder, "*.csv"))
    if not temp_csv_files:
        return

    # Concatenate all CSV files
    combined_df = pd.concat([pd.read_csv(f) for f in temp_csv_files], ignore_index=True)
    combined_df.to_csv(output_file, index=False)

def parse_arguments():
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process Geolife Trajectories data.")
    parser.add_argument(
        "--data-folder",
        type=str,
        default="Geolife Trajectories 1.3/Data",
        help="Path to the main data folder containing all user directories."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="all_users_data.csv",
        help="Path to the final consolidated CSV file."
    )
    parser.add_argument(
        "--temp-folder",
        type=str,
        default="temp_user_csvs",
        help="Path to the temporary folder for storing individual user CSV files."
    )
    return parser.parse_args()

def main():
    """
    Main function to process all users in parallel and write their data directly to a consolidated CSV file.
    """
    start_time = time.perf_counter()  # Record the start time

    # Parse command-line arguments
    args = parse_arguments()
    data_folder = args.data_folder
    output_file = args.output_file
    temp_folder = args.temp_folder

    # Initialize the temporary folder
    initialize_temp_folder(temp_folder)

    # Get a list of all user directories
    user_dirs = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    total_users = len(user_dirs)

    if total_users == 0:
        print("No user directories found. Exiting.")
        return

    # Prepare arguments for multiprocessing
    args_list = [(os.path.join(data_folder, user_dir), user_dir, temp_folder) for user_dir in user_dirs]

    # Use multiprocessing Pool to process users in parallel
    cpu_count = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=cpu_count) as pool:
        # Use tqdm to display progress bar for multiprocessing
        results = list(tqdm(pool.imap_unordered(process_user, args_list), total=total_users, desc="Processing users"))

    # Filter out any None results (users that were skipped)
    successful_csvs = [res for res in results if res is not None]

    # Concatenate all successful CSV files into the final output CSV
    concatenate_csv_files(temp_folder, output_file)

    end_time = time.perf_counter()  # Record the end time
    total_time = end_time - start_time  # Calculate total duration

    # Save the total processing time to a log file
    log_file = 'processing_time.log'
    with open(log_file, 'a') as f:
        f.write(f"Processing completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total processing time: {total_time:.2f} seconds.\n\n")

    # Clean up the temporary folder
    shutil.rmtree(temp_folder)

if __name__ == "__main__":
    main()
