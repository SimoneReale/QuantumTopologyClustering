import matplotlib.pyplot as plt
from create_dataset import haversine
import numpy as np
import seaborn as sns

def calculate_average_distances(df_taxi, df_5g, medoids_list):
    """
    Calculates the average of the distances from all the demand points to their nearest service point.

    Parameters:
    - df_taxi: DataFrame containing the taxi data (demand points)
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - medoids_list: List of selected medoids (indices of df_5g)

    Returns:
    - avg_distance: The average of the distances from all the demand points to their nearest service point
    - all_distances: List of distances from all the demand points to their nearest service point
    """
    all_distances = []
    for i, taxi in df_taxi.iterrows():
        min_distance = float('inf')
        for medoid in medoids_list:
            service_point = df_5g.iloc[medoid]
            distance = haversine(taxi['lat'], taxi['lon'], service_point['lat'], service_point['lon'])
            if distance < min_distance:
                min_distance = distance
        all_distances.append(min_distance)
    avg_distance = np.mean(all_distances)
    max_distance = np.max(all_distances)
    min_distance = np.min(all_distances)
    std_distance = np.std(all_distances)
    return max_distance, min_distance, avg_distance, std_distance, all_distances

def calculate_min_max_ratio(df_taxi, df_5g, medoids_list):
    """
    Calculates the min-max medoid distance ratio.

    Parameters:
    - df_taxi: DataFrame containing the taxi data (demand points)
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - medoids_list: List of selected medoids (indices of df_5g)

    Returns:
    - min_max_ratio: The min-max medoid distance ratio
    """
    min_distance = float('inf')
    max_distance = float('-inf')
    for i, taxi in df_taxi.iterrows():
        for medoid in medoids_list:
            service_point = df_5g.iloc[medoid]
            distance = haversine(taxi['lat'], taxi['lon'], service_point['lat'], service_point['lon'])
            if distance < min_distance:
                min_distance = distance
            if distance > max_distance:
                max_distance = distance
    if max_distance > 0:
        min_max_ratio = min_distance / max_distance
    else:
        min_max_ratio = None
    return min_max_ratio

def plot_boxplot(all_distances_list, labels):
    """
    Generates a box plot of the distances from all the demand points to their nearest service point for all solutions combined.

    Parameters:
    - all_distances_list: List of lists of distances for all solutions combined
    - labels: List of labels for the solutions
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=all_distances_list)
    plt.title('Box Plot of Distances from Demand Points to Nearest Service Point')
    plt.xlabel('Solutions')
    plt.ylabel('Distances')
    plt.xticks(ticks=range(len(labels)), labels=labels)

def plot_min_max_ratio_boxplot(min_max_ratio_list, labels):
    """
    Generates a box plot of the min-max medoid distance ratio for all solutions combined.

    Parameters:
    - min_max_ratio_list: List of min-max ratios for all solutions combined
    - labels: List of labels for the solutions
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=min_max_ratio_list)
    plt.title('Box Plot of Min-Max Medoid Distance Ratio')
    plt.xlabel('Solutions')
    plt.ylabel('Min-Max Ratio')
    plt.xticks(ticks=range(len(labels)), labels=labels)

