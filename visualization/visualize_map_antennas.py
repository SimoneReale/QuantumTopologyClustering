import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import folium
from folium.plugins import HeatMap

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









