import folium
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from folium.plugins import HeatMap
from create_dataset import create_pechino_dataset, create_reduced_dataset, haversine


def draw_chart_obj_fun_medoids(selected_medoids_list, compute_objective, Delta, df_taxi, df_5g, radius, figure_title="Objective Function Values"):
    """
    Plots separate charts for the values of dispersion_val + centrality_val + importance_val, dispersion_val, centrality_val, importance_val, constraint value, total distance between medoids, shortest distance between any medoid, and number of uncovered taxi calls.
    
    Parameters:
    - selected_medoids_list: List of lists, where each sublist contains a different set of selected medoids
    - compute_objective: Function to compute the objective values
    - Delta: Similarity matrix
    - df_taxi: DataFrame containing the taxi data
    - radius: Coverage radius of each medoid in meters
    - figure_title: Title of the figure
    """
    obj_values = []
    dispersion_values = []
    centrality_values = []
    importance_values = []
    constraint_values = []
    total_distance_values = []
    shortest_distance_values = []
    uncovered_taxi_calls_values = []
    uncovered_taxi_calls_percentage_values = []

    for selected_medoids in selected_medoids_list:
        dispersion_val, centrality_val, importance_val = compute_objective(selected_medoids)
        obj_values.append(dispersion_val + centrality_val + importance_val)
        dispersion_values.append(dispersion_val)
        centrality_values.append(centrality_val)
        importance_values.append(importance_val)
        constraint_values.append(len(selected_medoids))  # Constraint value is the number of selected medoids

        # Calculate total distance between selected medoids
        total_distance = sum(Delta[i, j] for i in selected_medoids for j in selected_medoids if i != j)
        total_distance_values.append(total_distance)

        # Calculate shortest distance between any medoid
        shortest_distance = min(Delta[i, j] for i in selected_medoids for j in selected_medoids if i != j)
        shortest_distance_values.append(shortest_distance)

        # Calculate number of uncovered taxi calls
        uncovered_taxi_calls = 0
        for _, taxi in df_taxi.iterrows():
            covered = False
            for medoid in selected_medoids:
                if haversine(taxi["lat"], taxi["lon"], df_5g.iloc[medoid]["lat"], df_5g.iloc[medoid]["lon"]) <= radius:
                    covered = True
                    break
            if not covered:
                uncovered_taxi_calls += 1
        uncovered_taxi_calls_values.append(uncovered_taxi_calls)
        uncovered_taxi_calls_percentage_values.append(uncovered_taxi_calls / len(df_taxi) * 100)

    # Create multiple figures to avoid clutter
    num_charts = 9
    charts_per_figure = 3
    num_figures = (num_charts + charts_per_figure - 1) // charts_per_figure

    for fig_idx in range(num_figures):
        fig, axes = plt.subplots(charts_per_figure, 1, figsize=(10, 18))
        fig.canvas.manager.set_window_title(f"{figure_title} - Part {fig_idx + 1}")
        fig.suptitle(f"{figure_title} - Part {fig_idx + 1}", fontsize=16)

        start_idx = fig_idx * charts_per_figure
        end_idx = min(start_idx + charts_per_figure, num_charts)

        for chart_idx in range(start_idx, end_idx):
            ax = axes[chart_idx - start_idx]
            if chart_idx == 0:
                ax.plot(obj_values, label='Objective Value (Dispersion + Centrality + Importance)', marker='o')
                ax.set_title('Objective Value (Dispersion + Centrality + Importance)')
                ax.set_xlabel('Solution Index')
                ax.set_ylabel('Value')
                ax.legend()
                ax.yaxis.set_major_locator(FixedLocator(obj_values))
            elif chart_idx == 1:
                ax.plot(dispersion_values, label='Dispersion Value', marker='o')
                ax.set_title('Dispersion Value')
                ax.set_xlabel('Solution Index')
                ax.set_ylabel('Value')
                ax.legend()
                ax.yaxis.set_major_locator(FixedLocator(dispersion_values))
            elif chart_idx == 2:
                ax.plot(centrality_values, label='Centrality Value', marker='o')
                ax.set_title('Centrality Value')
                ax.set_xlabel('Solution Index')
                ax.set_ylabel('Value')
                ax.legend()
                ax.yaxis.set_major_locator(FixedLocator(centrality_values))
            elif chart_idx == 3:
                ax.plot(importance_values, label='Importance Value', marker='o')
                ax.set_title('Importance Value')
                ax.set_xlabel('Solution Index')
                ax.set_ylabel('Value')
                ax.legend()
                ax.yaxis.set_major_locator(FixedLocator(importance_values))
            elif chart_idx == 4:
                ax.plot(constraint_values, label='Constraint Value (Number of Selected Medoids)', marker='o')
                ax.set_title('Constraint Value (Number of Selected Medoids)')
                ax.set_xlabel('Solution Index')
                ax.set_ylabel('Value')
                ax.legend()
                ax.yaxis.set_major_locator(FixedLocator(constraint_values))
            elif chart_idx == 5:
                ax.plot(total_distance_values, label='Total Distance Between Medoids', marker='o')
                ax.set_title('Total Distance Between Medoids')
                ax.set_xlabel('Solution Index')
                ax.set_ylabel('Value')
                ax.legend()
                ax.yaxis.set_major_locator(FixedLocator(total_distance_values))
            elif chart_idx == 6:
                ax.plot(shortest_distance_values, label='Shortest Distance Between Any Medoid', marker='o')
                ax.set_title('Shortest Distance Between Any Medoid')
                ax.set_xlabel('Solution Index')
                ax.set_ylabel('Value')
                ax.legend()
                ax.yaxis.set_major_locator(FixedLocator(shortest_distance_values))
            elif chart_idx == 7:
                ax.plot(uncovered_taxi_calls_values, label='Uncovered Taxi Calls (Absolute)', marker='o')
                ax.set_title('Uncovered Taxi Calls (Absolute)')
                ax.set_xlabel('Solution Index')
                ax.set_ylabel('Number of Uncovered Taxi Calls')
                ax.legend()
                ax.yaxis.set_major_locator(FixedLocator(uncovered_taxi_calls_values))
            elif chart_idx == 8:
                ax.plot(uncovered_taxi_calls_percentage_values, label='Uncovered Taxi Calls (Percentage)', marker='o')
                ax.set_title('Uncovered Taxi Calls (Percentage)')
                ax.set_xlabel('Solution Index')
                ax.set_ylabel('Percentage of Uncovered Taxi Calls')
                ax.legend()
                ax.yaxis.set_major_locator(FixedLocator(uncovered_taxi_calls_percentage_values))

        plt.tight_layout(rect=[0, 0, 1, 0.96])