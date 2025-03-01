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

def draw_chart_obj_fun_medoids(selected_medoids_list, compute_objective, figure_title="Objective Function Values"):
    """
    Plots separate charts for the values of dispersion_val + centrality_val + importance_val, dispersion_val, centrality_val, and importance_val.
    
    Parameters:
    - selected_medoids_list: List of lists, where each sublist contains a different set of selected medoids
    - compute_objective: Function to compute the objective values
    - figure_title: Title of the figure
    """
    obj_values = []
    dispersion_values = []
    centrality_values = []
    importance_values = []

    for selected_medoids in selected_medoids_list:
        dispersion_val, centrality_val, importance_val = compute_objective(selected_medoids)
        obj_values.append(dispersion_val + centrality_val + importance_val)
        dispersion_values.append(dispersion_val)
        centrality_values.append(centrality_val)
        importance_values.append(importance_val)

    fig, axes = plt.subplots(4, 1, figsize=(10, 24))
    fig.canvas.manager.set_window_title(figure_title)
    fig.suptitle(figure_title, fontsize=16)

    axes[0].plot(obj_values, label='Objective Value (Dispersion + Centrality + Importance)', marker='o')
    axes[0].set_title('Objective Value (Dispersion + Centrality + Importance)')
    axes[0].set_xlabel('Solution Index')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].yaxis.set_major_locator(FixedLocator(obj_values))

    axes[1].plot(dispersion_values, label='Dispersion Value', marker='o')
    axes[1].set_title('Dispersion Value')
    axes[1].set_xlabel('Solution Index')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].yaxis.set_major_locator(FixedLocator(dispersion_values))

    axes[2].plot(centrality_values, label='Centrality Value', marker='o')
    axes[2].set_title('Centrality Value')
    axes[2].set_xlabel('Solution Index')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].yaxis.set_major_locator(FixedLocator(centrality_values))

    axes[3].plot(importance_values, label='Importance Value', marker='o')
    axes[3].set_title('Importance Value')
    axes[3].set_xlabel('Solution Index')
    axes[3].set_ylabel('Value')
    axes[3].legend()
    axes[3].yaxis.set_major_locator(FixedLocator(importance_values))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

def plot_medoids_on_map(df_5g, df_taxi, selected_medoids, map_title="Selected Medoids"):
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
            popup=f"Cell ID: {row['cell']} | Signal: {row.get('averageSignal', 'N/A')}",
        ).add_to(map_5g)

    for medoid in selected_medoids:
        row = df_5g.iloc[int(medoid)]
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"Medoid Cell ID: {row['cell']} | Signal: {row.get('averageSignal', 'N/A')}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(map_5g)

    heat_data = list(zip(df_taxi["lat"], df_taxi["lon"]))
    HeatMap(heat_data).add_to(map_5g)

    map_5g.save(f"{map_title}.html")







