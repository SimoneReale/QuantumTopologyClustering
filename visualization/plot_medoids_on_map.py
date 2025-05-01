import folium
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from folium.plugins import HeatMap
from create_dataset import create_pechino_dataset, create_reduced_dataset, haversine


def plot_medoids_on_map(df_5g, df_taxi, selected_medoids, Delta, coverage_radius, map_title="Selected Medoids", plot_radius=True, plot_distances = True, filename = "folium_output/placeholder.html"):
    """
    Plots the selected medoids on a map, draws coverage radius circles, 
    and connects them with lines displaying distances. Also plots the location of taxi calls that are not served by any antenna.

    Parameters:
    - df_5g: DataFrame containing the 5G antenna data.
    - df_taxi: DataFrame containing the taxi data.
    - selected_medoids: List of selected medoids.
    - Delta: Distance matrix.
    - coverage_radius: Radius of coverage for each medoid in meters.
    - map_title: Title of the map.
    - plot_radius: Boolean indicating whether to plot the coverage radius circles.
    """
    map_center = [df_5g["lat"].mean(), df_5g["lon"].mean()]
    map_5g = folium.Map(location=map_center, zoom_start=12)

    # Recompute taxi_count for selected medoids
    medoid_taxi_counts = {medoid: 0 for medoid in selected_medoids}
    for _, taxi in df_taxi.iterrows():
        nearest_medoid = None
        min_distance = float('inf')
        for medoid in selected_medoids:
            distance = haversine(taxi["lat"], taxi["lon"], df_5g.iloc[medoid]["lat"], df_5g.iloc[medoid]["lon"])
            if distance < min_distance:
                min_distance = distance
                nearest_medoid = medoid
        if nearest_medoid is not None:
            medoid_taxi_counts[nearest_medoid] += 1

    # Plot selected medoids with coverage radius and updated taxi counts
    for medoid in selected_medoids:
        row = df_5g.iloc[int(medoid)]
        
        # Medoid marker
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"Medoid Cell ID: {row['cell']} | Importance: {row['importance']} | Taxi Count: {medoid_taxi_counts[medoid]}",
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
    if plot_distances:
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

    # Plot the location of taxi calls that are not served by any antenna
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
    map_5g.save(filename)