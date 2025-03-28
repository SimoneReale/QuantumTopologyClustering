import os
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np

def create_taxi_splits_with_edge_calls(
    folder_path, output_folder, num_taxis, time_interval_hours, num_splits, avg_calls_per_taxi_per_hour
):
    """
    Creates personalized splits of the taxi dataset, keeping only rows where an edge function call occurs.

    Parameters:
    folder_path (str): Path to the folder containing taxi log files.
    output_folder (str): Path to the folder where split files will be saved.
    num_taxis (int): Number of taxis to include in the splits.
    time_interval_hours (int): Time interval in hours for each split.
    num_splits (int): Number of splits to generate.
    avg_calls_per_taxi_per_hour (float): Average number of edge function calls per taxi per hour.

    Returns:
    None
    """
    # Get all taxi files in the folder
    taxi_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    # Select a random subset of taxis
    selected_taxis = random.sample(taxi_files, min(num_taxis, len(taxi_files)))
    
    # Read data from selected taxis
    data = []
    for taxi_file in selected_taxis:
        file_path = os.path.join(folder_path, taxi_file)
        df = pd.read_csv(file_path, names=["taxi_id", "datetime", "longitude", "latitude"], parse_dates=["datetime"])
        data.append(df)
    
    # Concatenate all selected taxi data
    all_data = pd.concat(data, ignore_index=True)
    
    # Sort by datetime
    all_data = all_data.sort_values(by="datetime")
    
    # Determine the start and end times for the splits
    start_time = all_data["datetime"].min()
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate and save splits
    for i in range(num_splits):
        split_start = start_time + i * timedelta(hours=time_interval_hours)
        split_end = split_start + timedelta(hours=time_interval_hours)
        split_data = all_data[(all_data["datetime"] >= split_start) & (all_data["datetime"] < split_end)]
        
        # Assign edge calls based on a Poisson process for each taxi
        def assign_edge_calls(df):
            taxi_id = df["taxi_id"].iloc[0]
            lambda_calls = avg_calls_per_taxi_per_hour * time_interval_hours
            num_calls = np.random.poisson(lambda_calls)  # Number of calls for this taxi
            call_indices = np.random.choice(len(df), min(num_calls, len(df)), replace=False)
            return df.iloc[call_indices]
        
        edge_call_data = split_data.groupby("taxi_id", group_keys=False).apply(assign_edge_calls)
        
        # Save split to file
        split_filename = os.path.join(output_folder, f"split_{i+1}.txt")
        edge_call_data.to_csv(split_filename, index=False, header=False)
    
    print(f"{num_splits} splits with edge calls saved in {output_folder}")

if __name__ == "__main__":
    create_taxi_splits_with_edge_calls("./complete_taxi_data", "splits", 100, 8, 1, avg_calls_per_taxi_per_hour=20)
