import folium
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from folium.plugins import HeatMap
from create_dataset import create_pechino_dataset, create_reduced_dataset, haversine
import numpy as np
import seaborn as sns

def calculate_average_distances(df_taxi, df_5g, selected_medoids):
    """
    Calculates the average of the averages of the distances from all the demand points to their nearest service point.

    Parameters:
    - df_taxi: DataFrame containing the taxi data (demand points)
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - selected_medoids: List of selected medoids (indices of df_5g)

    Returns:
    - avg_of_avgs: The average of the averages of the distances
    """
    distances = []

    for i, taxi in df_taxi.iterrows():
        min_distance = float('inf')
        for medoid in selected_medoids:
            service_point = df_5g.iloc[medoid]
            distance = haversine(taxi['lat'], taxi['lon'], service_point['lat'], service_point['lon'])
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)

    avg_of_avgs = np.mean(distances)
    return avg_of_avgs, distances

def plot_boxplot(distances):
    """
    Generates a box plot of the distances from all the demand points to their nearest service point.

    Parameters:
    - distances: List of distances from all the demand points to their nearest service point
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=distances)
    plt.title('Box Plot of Distances from Demand Points to Nearest Service Point')
    plt.xlabel('Distances')
    plt.ylabel('Value')
    plt.show()






