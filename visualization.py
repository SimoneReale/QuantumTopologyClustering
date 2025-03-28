import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import folium
from folium.plugins import HeatMap
from create_dataset import create_pechino_dataset, create_reduced_dataset, haversine

def visualize_graph(n, Delta, points, selected_medoids_list, compute_objective, figure_title="Graph Visualization"):
    """
    Plots multiple graphs for different sets of selected medoids in a single figure.
    
    Parameters:
    - n: Number of nodes
    - Delta: Similarity matrix
    - points: 2D coordinates of points
    - selected_medoids_list: List of lists, where each sublist contains a different set of selected medoids
    - compute_objective: Function to compute the objective values
    - figure_title: Title of the figure
    """
    num_graphs = len(selected_medoids_list)
    cols = min(num_graphs, 2)  
    rows = (num_graphs + 1) // 2

    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    fig.canvas.manager.set_window_title(figure_title)
    fig.suptitle(figure_title, fontsize=16)
    axes = fig.subplots(rows, cols)
    axes = axes.flatten() if num_graphs > 1 else [axes] 

    for idx, selected_medoids in enumerate(selected_medoids_list):
        ax = axes[idx]
        G = nx.Graph()

        dispersion_val, centrality_val, importance_val = compute_objective(selected_medoids)

        for i in range(n):
            G.add_node(i, pos=(points[i, 0], points[i, 1]))

        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=Delta[i, j])

        pos = nx.get_node_attributes(G, 'pos')

        nx.draw(G, pos, with_labels=True, 
                node_color=['red' if i in selected_medoids else 'blue' for i in range(n)], 
                node_size=300, ax=ax)

        nx.draw_networkx_edge_labels(G, pos, 
                                     edge_labels={(i, j): f"{Delta[i, j]:.2f}" for i, j in G.edges()}, 
                                     font_size=7, ax=ax)

        ax.set_title(f"Sol {idx + 1} | Obj: {dispersion_val + centrality_val + importance_val:.4f} | Disp: {dispersion_val:.4f} | Cen: {centrality_val:.4f} | Imp: {importance_val:.4f}")

    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()

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


def plot_medoids_on_map(df_5g, df_taxi, selected_medoids, Delta, coverage_radius, map_title="Selected Medoids", plot_radius=True):
    """
    Plots the selected medoids on a map, draws coverage radius circles, 
    and connects them with lines displaying distances.

    Parameters:
    - df_5g: DataFrame containing the 5G antenna data.
    - df_taxi: DataFrame containing the taxi data.
    - selected_medoids: List of selected medoids.
    - Delta: Distance matrix.
    - coverage_radius: Radius of coverage for each medoid in meters.
    - map_title: Title of the map.
    """
    map_center = [df_5g["lat"].mean(), df_5g["lon"].mean()]
    map_5g = folium.Map(location=map_center, zoom_start=12)

    # Plot all 5G antenna locations
    for _, row in df_5g.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup=f"Cell ID: {row['cell']} | Importance: {row['importance']} | Taxi Count: {row['taxi_count']}",
        ).add_to(map_5g)

    # Plot selected medoids with coverage radius
    for medoid in selected_medoids:
        row = df_5g.iloc[int(medoid)]
        
        # Medoid marker
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"Medoid Cell ID: {row['cell']} | Importance: {row['importance']} | Taxi Count: {row['taxi_count']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(map_5g)

        # Coverage radius circle
        if plot_radius:
            folium.Circle(
                location=[row["lat"], row["lon"]],
                radius=coverage_radius,  # Coverage radius in meters
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.2
            ).add_to(map_5g)


    # Draw lines between selected medoids and add distances
    for i in range(len(selected_medoids)):
        for j in range(i + 1, len(selected_medoids)):
            medoid1 = int(selected_medoids[i])
            medoid2 = int(selected_medoids[j])

            lat_lon1 = [df_5g.iloc[medoid1]["lat"], df_5g.iloc[medoid1]["lon"]]
            lat_lon2 = [df_5g.iloc[medoid2]["lat"], df_5g.iloc[medoid2]["lon"]]
            distance = Delta[medoid1, medoid2]

            # Draw connection line
            folium.PolyLine(
                locations=[lat_lon1, lat_lon2],
                color="green",
                weight=2.5,
                opacity=1,
                popup=f"Distance: {distance:.2f} meters"
            ).add_to(map_5g)

            # Add distance label
            mid_lat = (lat_lon1[0] + lat_lon2[0]) / 2
            mid_lon = (lat_lon1[1] + lat_lon2[1]) / 2
            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(html=f'<div style="font-size: 12px; color: green;">{distance:.2f} m</div>')
            ).add_to(map_5g)

    for _, taxi in df_taxi.iterrows():
        covered = False
        for medoid in selected_medoids:
            if haversine(taxi["lat"], taxi["lon"], df_5g.iloc[medoid]["lat"], df_5g.iloc[medoid]["lon"]) <= coverage_radius:
                covered = True
                break
        if not covered:
            folium.CircleMarker(
                location=[taxi["lat"], taxi["lon"]],
                radius=5,
                color="black",
                fill=True,
                fill_color="black",
                fill_opacity=0.6,
                popup=f"Uncovered Taxi Call",
            ).add_to(map_5g)

    # Add heatmap for taxi data
    heat_data = list(zip(df_taxi["lat"], df_taxi["lon"]))
    HeatMap(heat_data).add_to(map_5g)

    # Save the map
    map_5g.save(f"folium_output/{map_title}.html")


def visualize_map_antennas_heatmap(df_5g, df_taxi, coverage_radius = 0, map_title="Antennas and Taxi Heatmap"):
    """
    Plots the antennas with the taxi count and the heatmap.

    Parameters:
    - df_5g: DataFrame containing the 5G antenna data
    - df_taxi: DataFrame containing the taxi data
    - map_title: Title of the map
    """
    map_center = [df_5g["lat"].mean(), df_5g["lon"].mean()]
    map_5g = folium.Map(location=map_center, zoom_start=12)

    for _, row in df_5g.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup=f"Cell ID: {row['cell']} | Taxi Count: {row['taxi_count']}",
        ).add_to(map_5g)

    if coverage_radius != 0:
        # Coverage radius circle
            folium.Circle(
                location=[row["lat"], row["lon"]],
                radius=coverage_radius,  # Coverage radius in meters
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.2
            ).add_to(map_5g)

    heat_data = list(zip(df_taxi["lat"], df_taxi["lon"]))
    HeatMap(heat_data).add_to(map_5g)

    map_5g.save(f"folium_output/{map_title}.html")

if __name__ == "__main__":
    Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(N_CLUSTERS=80, filter_radius=6000)
    visualize_map_antennas_heatmap(df_5g, df_taxi, map_title="Antennas and Taxi Heatmap")







