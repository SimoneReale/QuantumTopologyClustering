import numpy as np
from create_dataset import haversine, create_pechino_dataset
import random

def assign_taxi_calls_to_nearest_medoids(df_taxi, df_5g, selected_medoids):
    """
    Assigns each taxi call to the nearest medoid.

    Parameters:
    - df_taxi: DataFrame containing the taxi data (demand points)
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - selected_medoids: List of selected medoids (indices of df_5g)

    Returns:
    - assignments: List of assignments where each element is the index of the medoid assigned to the corresponding taxi call
    """
    assignments = []
    for _, taxi in df_taxi.iterrows():
        min_distance = float('inf')
        assigned_medoid = None
        for medoid in selected_medoids:
            service_point = df_5g.iloc[medoid]
            distance = haversine(taxi['lat'], taxi['lon'], service_point['lat'], service_point['lon'])
            if distance < min_distance:
                min_distance = distance
                assigned_medoid = medoid
        assignments.append(assigned_medoid)
    return assignments

def assign_taxi_calls_to_random_medoids(df_taxi, df_5g, selected_medoids, radius):
    """
    Assigns each taxi call to a random medoid if the taxi call is within the radius of that medoid.

    Parameters:
    - df_taxi: DataFrame containing the taxi data (demand points)
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - selected_medoids: List of selected medoids (indices of df_5g)
    - radius: Radius within which a taxi call is considered to be covered by a medoid

    Returns:
    - assignments: List of assignments where each element is the index of the medoid assigned to the corresponding taxi call
    """
    assignments = []
    for _, taxi in df_taxi.iterrows():
        possible_medoids = []
        for medoid in selected_medoids:
            service_point = df_5g.iloc[medoid]
            distance = haversine(taxi['lat'], taxi['lon'], service_point['lat'], service_point['lon'])
            if distance <= radius:
                possible_medoids.append(medoid)
        if possible_medoids:
            assigned_medoid = random.choice(possible_medoids)
        else:
            assigned_medoid = None
        assignments.append(assigned_medoid)
    return assignments

def calculate_fairness_and_equity_metrics(df_taxi, df_5g, selected_medoids, filter_radius, assign_type="nearest"):
    """
    Calculates fairness and equity metrics for the distribution of taxi calls to medoids.

    Parameters:
    - df_taxi: DataFrame containing the taxi data (demand points)
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - selected_medoids: List of selected medoids (indices of df_5g)
    - filter_radius: Radius within which a taxi call is considered to be covered by a medoid
    - assign_type: Type of assignment ("nearest" or "radius")

    Returns:
    - max_demand_per_medoid: Maximum number of taxi calls assigned to any medoid
    - demand_to_medoid_ratio_variance: Variance in the number of taxi calls assigned to each medoid
    """
    # Assign taxi calls to medoids
    if assign_type == "nearest":
        assignments = assign_taxi_calls_to_nearest_medoids(df_taxi, df_5g, selected_medoids)
    else:
        assignments = assign_taxi_calls_to_random_medoids(df_taxi, df_5g, selected_medoids, filter_radius)

    # Filter out unassigned taxi calls
    assignments = [medoid for medoid in assignments if medoid is not None]

    # Calculate the number of taxi calls assigned to each medoid
    demand_counts = np.zeros(len(selected_medoids))
    for medoid in assignments:
        demand_counts[selected_medoids.index(medoid)] += 1

    # Calculate the maximum demand per medoid
    max_demand_per_medoid = np.max(demand_counts)

    # Calculate the average demand per medoid
    avg_demand_per_medoid = np.mean(demand_counts)

    # Calculate the variance in the number of taxi calls assigned to each medoid
    demand_to_medoid_ratio_std = np.std(demand_counts)

    return max_demand_per_medoid, demand_to_medoid_ratio_std, avg_demand_per_medoid



