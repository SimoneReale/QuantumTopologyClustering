import folium
from scipy.spatial import ConvexHull
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

def plot_convex_hull(df_5g, selected_medoids, filename = "folium_output/convex_hull_map_bqm.html"):
    """
    Calculates the convex hull created by the selected medoids and plots it with Folium.

    Parameters:
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - selected_medoids: List of selected medoids (indices of df_5g)

    Returns:
    - m: Folium map with the convex hull and selected medoids plotted
    - area_km2: Area of the convex hull in square kilometers
    """
    # Extract the coordinates of the selected medoids
    points = df_5g.iloc[selected_medoids][['lat', 'lon']].values

    # Calculate the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Ensure the hull is closed by adding the first point to the end
    hull_points = np.append(hull_points, [hull_points[0]], axis=0)

    # Create a Folium map centered around the mean location of the selected medoids
    center_lat = np.mean(hull_points[:, 0])
    center_lon = np.mean(hull_points[:, 1])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Add the convex hull to the map
    folium.PolyLine(locations=hull_points, color='blue').add_to(m)

    # Add the selected medoids to the map
    for point in points:
        folium.Marker(location=[point[0], point[1]], icon=folium.Icon(color='red')).add_to(m)

    # Calculate the area of the convex hull using the shoelace formula
    x = hull_points[:, 1]
    y = hull_points[:, 0]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    area_km2 = area * (111 ** 2)  # Convert from degrees to square kilometers

    m.save(filename)

    return area_km2

def calculate_medoid_density(selected_medoids, area_km2):
    """
    Calculates the medoid density per square kilometer.

    Parameters:
    - selected_medoids: List of selected medoids (indices of df_5g)
    - area_km2: Area of the convex hull in square kilometers

    Returns:
    - density: Medoid density per square kilometer
    """
    # Calculate the medoid density
    density = len(selected_medoids) / area_km2

    return density