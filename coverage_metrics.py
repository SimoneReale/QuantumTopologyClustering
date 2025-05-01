import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from create_dataset import haversine

def calculate_coverage(df_taxi, df_5g, medoids_list, coverage_radius):
    """
    Calculates the mean percentage coverage of taxi calls for each solution.

    Parameters:
    - df_taxi: DataFrame containing the taxi data (demand points)
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - medoids_list: List of selected medoids (indices of df_5g)
    - coverage_radius: Radius within which a taxi call is considered covered

    Returns:
    - coverage_percentage: Coverage percentage of taxi calls for the given medoids
    """
    covered_taxi_calls = 0
    for i, taxi in df_taxi.iterrows():
        for medoid in medoids_list:
            service_point = df_5g.iloc[medoid]
            distance = haversine(taxi['lat'], taxi['lon'], service_point['lat'], service_point['lon'])
            if distance <= coverage_radius:
                covered_taxi_calls += 1
                break
    coverage_percentage = (covered_taxi_calls / len(df_taxi)) * 100
    return coverage_percentage

def plot_coverage_boxplot(coverage_percentages, labels):
    """
    Generates a box plot of the coverage percentages for all solutions combined.

    Parameters:
    - coverage_percentages: List of coverage percentages for all solutions combined
    - labels: List of labels for the solutions
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=coverage_percentages)
    plt.title('Box Plot of Coverage Percentages')
    plt.xlabel('Solutions')
    plt.ylabel('Coverage Percentage')
    plt.xticks(ticks=range(len(labels)), labels=labels)

if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create example data
    data_taxi = {
        'lat': [41.9028, 41.9029, 41.9030, 41.9031, 41.9032],
        'lon': [12.4964, 12.4965, 12.4966, 12.4967, 12.4968]
    }
    data_5g = {
        'lat': [41.9028, 41.9030, 41.9032],
        'lon': [12.4964, 12.4966, 12.4968]
    }
    df_taxi = pd.DataFrame(data_taxi)
    df_5g = pd.DataFrame(data_5g)
    medoids_list = [0, 1, 2]  # Example selected medoids
    coverage_radius = 1.0  # Example coverage radius in kilometers

    # Calculate coverage percentage
    coverage_percentage = calculate_coverage(df_taxi, df_5g, medoids_list, coverage_radius)
    print(f"Coverage Percentage: {coverage_percentage}")

    # Example coverage percentages for multiple solutions
    coverage_percentages = [[coverage_percentage], [coverage_percentage - 5], [coverage_percentage + 5]]
    labels = ['Solution 1', 'Solution 2', 'Solution 3']

    # Plot boxplot of coverage percentages
    plot_coverage_boxplot(coverage_percentages, labels)
    plt.show()
