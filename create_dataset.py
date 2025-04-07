import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, euclidean
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from visualization.visualize_map_antennas import visualize_map_antennas_heatmap
import time  # Import the time module


def filter_taxi_calls(df_taxi, df_5g, radius):
    """
    Filter out taxi calls that don't have any antenna within a specified radius.

    Parameters:
    - df_taxi: DataFrame containing the taxi data
    - df_5g: DataFrame containing the 5G antenna data
    - radius: Radius in meters to check for nearby antennas

    Returns:
    - filtered_df_taxi: Filtered DataFrame containing only taxi calls with nearby antennas
    """
    taxi_tree = cKDTree(df_taxi[["lat", "lon"]].values)
    antenna_tree = cKDTree(df_5g[["lat", "lon"]].values)
    indices = taxi_tree.query_ball_tree(
        antenna_tree, radius / 111320
    )  # Convert radius to degrees
    valid_indices = [i for i, nearby_antennas in enumerate(indices) if nearby_antennas]
    filtered_df_taxi = df_taxi.iloc[valid_indices].reset_index(drop=True)
    return filtered_df_taxi


# Calculate distance matrix in meters using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth specified in decimal degrees.
    Returns the distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of Earth in meters
    return c * r


def calculate_importance_standard(df_5g):
    """
    Calculate importance values using the standard method.

    Parameters:
    - df_5g: DataFrame containing the 5G antenna data

    Returns:
    - importance_values: Normalized importance values
    """
    importance_values = df_5g["taxi_count"].values
    importance_values = 1 * (
        importance_values / importance_values.max()
    )  # Normalize to [0, 1]
    return importance_values


def calculate_importance_radius(df_5g, df_taxi, radius):
    """
    Calculate importance values based on the number of taxis within a specified radius for each antenna.

    Parameters:
    - df_5g: DataFrame containing the 5G antenna data
    - df_taxi: DataFrame containing the taxi data
    - radius: Radius in meters to count the number of taxis

    Returns:
    - importance_values: Normalized importance values
    """
    importance_values = np.zeros(len(df_5g))
    antenna_tree = cKDTree(df_5g[["lat", "lon"]].values)
    taxi_tree = cKDTree(df_taxi[["lat", "lon"]].values)

    for idx, antenna in enumerate(df_5g.itertuples(index=False)):
        taxis_within_radius = taxi_tree.query_ball_point(
            [antenna.lat, antenna.lon], radius / 111320
        )  # Convert radius to degrees
        importance_values[idx] = len(taxis_within_radius)

    importance_values = 1 * (
        importance_values / importance_values.max()
    )  # Normalize to [0, 1]
    return importance_values


def calculate_combined_importance(df_5g, df_taxi, radius):
    """
    Calculate combined importance values based on the number of taxis within a specified radius and unique coverage of demand.

    Parameters:
    - df_5g: DataFrame containing the 5G antenna data
    - df_taxi: DataFrame containing the taxi data
    - radius: Radius in meters to count the number of taxis

    Returns:
    - importance_values: Combined normalized importance values
    """
    # Calculate radius-based importance
    radius_importance = calculate_importance_radius(df_5g, df_taxi, radius)

    # Calculate unique coverage-based importance
    coverage_importance = np.zeros(len(df_5g))
    taxi_tree = cKDTree(df_taxi[["lat", "lon"]].values)
    antenna_tree = cKDTree(df_5g[["lat", "lon"]].values)

    for idx, antenna in enumerate(df_5g.itertuples(index=False)):
        taxis_within_radius = taxi_tree.query_ball_point(
            [antenna.lat, antenna.lon], radius / 111320
        )  # Convert radius to degrees
        for taxi_idx in taxis_within_radius:
            taxi_location = df_taxi.iloc[taxi_idx][["lat", "lon"]].values
            covering_antennas = antenna_tree.query_ball_point(
                taxi_location, radius / 111320
            )
            if len(covering_antennas) == 1:
                coverage_importance[idx] += 1

    # Normalize coverage importance
    coverage_importance = 1 * (coverage_importance / coverage_importance.max())

    # Combine the two importance values
    combined_importance = radius_importance + 0.5 * coverage_importance
    combined_importance = 1 * (
        combined_importance / combined_importance.max()
    )  # Normalize to [0, 1]

    # print(coverage_importance)

    return combined_importance


def create_pechino_dataset(filter_radius=10000, is_importance_radius=True, taxi_data_file="taxi_data/taxi_data_8.txt"):
    MIN_LAT, MAX_LAT = 39.5, 41.0
    MIN_LON, MAX_LON = 115.5, 117.5

    df_taxi = pd.read_csv(
        taxi_data_file,
        header=None,
        names=["taxi_id", "datetime", "lon", "lat"],
    )

    file_path_antennas = "csv5G/460.csv"
    column_names = [
        "radio",
        "mcc",
        "net",
        "area",
        "cell",
        "unit",
        "lon",
        "lat",
        "range",
        "samples",
        "changeable",
        "created",
        "updated",
        "averageSignal",
    ]
    df_antennas = pd.read_csv(file_path_antennas, names=column_names, skiprows=1)

    df_5g_unfiltered = df_antennas[df_antennas["radio"].str.contains("LTE", na=False)]
    df_5g = df_5g_unfiltered[
        (df_5g_unfiltered["mcc"] == 460)
        & (df_5g_unfiltered["lat"].between(MIN_LAT, MAX_LAT))
        & (df_5g_unfiltered["lon"].between(MIN_LON, MAX_LON))
    ].copy()

    df_taxi = filter_taxi_calls(df_taxi, df_5g, filter_radius)

    antenna_tree = cKDTree(df_5g[["lat", "lon"]].values)
    _, nearest_antennas = antenna_tree.query(df_taxi[["lat", "lon"]].values)

    taxi_count_per_antenna = np.bincount(nearest_antennas, minlength=len(df_5g))
    df_5g["taxi_count"] = taxi_count_per_antenna
    df_5g = df_5g[df_5g["taxi_count"] > 0]

    print("Number of antennas: ", len(df_5g))

    dist_matrix = np.zeros((len(df_5g), len(df_5g)))
    for i in range(len(df_5g)):
        for j in range(len(df_5g)):
            if i != j:
                dist_matrix[i, j] = haversine(
                    df_5g.iloc[i]["lat"],
                    df_5g.iloc[i]["lon"],
                    df_5g.iloc[j]["lat"],
                    df_5g.iloc[j]["lon"],
                )

    Delta = dist_matrix

    start_time = time.time()
    if is_importance_radius:
        importance_values = calculate_combined_importance(df_5g, df_taxi, filter_radius)
    else:
        importance_values = calculate_importance_standard(df_5g)

    end_time = time.time()
    dataset_creation_time = end_time - start_time

    print(f"Time taken to calculate importance values: {dataset_creation_time:.2f} seconds")

    df_5g["importance"] = importance_values

    return Delta, len(df_5g), df_5g, df_taxi, importance_values


def create_reduced_dataset(N_CLUSTERS, filter_radius=10000, is_importance_radius=True, taxi_data_file="taxi_data/taxi_data_8.txt"):
    Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(
        filter_radius, is_importance_radius=is_importance_radius, taxi_data_file=taxi_data_file
    )

    if(len(df_5g) <= N_CLUSTERS):
        return Delta, n, df_5g, df_taxi, importance_values

    coords = df_5g[["lat", "lon"]].to_numpy()
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(coords)
    centroids = kmeans.cluster_centers_

    selected_antennas = []
    for center in centroids:
        closest_idx = np.argmin(
            [
                haversine(center[0], center[1], lat, lon)
                for lat, lon in zip(df_5g["lat"], df_5g["lon"])
            ]
        )
        selected_antennas.append(df_5g.iloc[closest_idx])

    df_selected = pd.DataFrame(selected_antennas)

    dist_matrix_reduced = np.zeros((len(df_selected), len(df_selected)))
    for i in range(len(df_selected)):
        for j in range(len(df_selected)):
            if i != j:
                dist_matrix_reduced[i, j] = haversine(
                    df_selected.iloc[i]["lat"],
                    df_selected.iloc[i]["lon"],
                    df_selected.iloc[j]["lat"],
                    df_selected.iloc[j]["lon"],
                )

    Delta_reduced = dist_matrix_reduced  # Use the distance matrix directly in meters

    if is_importance_radius:
        importance_values_reduced = calculate_combined_importance(
            df_selected, df_taxi, filter_radius
        )
    else:
        importance_values_reduced = calculate_importance_standard(df_selected)
    df_selected["importance"] = importance_values_reduced

    return (
        Delta_reduced,
        len(df_selected),
        df_selected,
        df_taxi,
        importance_values_reduced,
    )


def plot_distance_spread(Delta, importance_values_combined, importance_values_radius):
    """
    Plots the spread of distances in the Delta matrix and the distributions of importance values.

    Parameters:
    - Delta: Distance matrix
    - importance_values_standard: Importance values calculated using the standard method
    - importance_values_radius: Importance values calculated using the radius method
    """
    distances = Delta[
        np.triu_indices_from(Delta, k=1)
    ]  # Extract upper triangular part of the matrix
    avg_distance = np.mean(distances)

    standard_deviation = np.std(distances)
    print(f"Standard deviation and mean of distances: {standard_deviation} {avg_distance}")

    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, edgecolor="black")
    plt.axvline(avg_distance, color="r", linestyle="dashed", linewidth=1)
    plt.text(avg_distance, plt.ylim()[1] * 0.9, f"Avg: {avg_distance:.2f}", color="r")
    plt.title("Spread of Distances in Delta Matrix")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.hist(importance_values_combined, bins=50, edgecolor="black")
    plt.title("Distribution of Importance Values (Combined Method)")
    plt.xlabel("Importance Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.hist(importance_values_radius, bins=50, edgecolor="black")
    plt.title("Distribution of Importance Values (Radius Method)")
    plt.xlabel("Importance Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    filter_radius = 8000
    split_name = "split_1"
    times = []
    # Start timing the dataset creation
    for _ in range(10):
        Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(
            filter_radius, taxi_data_file=f"splits/{split_name}.txt"
        )

    
    print(f"Average time: {np.mean(times):.2f} Standard deviation: {np.std(times):.2f}")

    #print(f"Time taken to create the dataset and calculate importance values: {dataset_creation_time:.2f} seconds")
    
    # Visualize the map
    # visualize_map_antennas_heatmap(
    #     df_5g, df_taxi, map_title=f"Antennas_and_Taxi_Heatmap_{split_name}"
    # )
