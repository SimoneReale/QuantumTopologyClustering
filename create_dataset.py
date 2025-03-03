import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, euclidean
from sklearn.cluster import KMeans

def create_pechino_dataset():
    MIN_LAT, MAX_LAT = 39.5, 41.0
    MIN_LON, MAX_LON = 115.5, 117.5

    df_taxi = pd.read_csv(
        "taxi_data/taxi_data_2.txt",
        header=None,
        names=["taxi_id", "datetime", "lon", "lat"],
    )

    file_path_antennas = "csv5G/460.csv"
    column_names = [
        "radio",
        "mcc",
        "net",
        "area",
        "cell",
        "unit",
        "lon",
        "lat",
        "range",
        "samples",
        "changeable",
        "created",
        "updated",
        "averageSignal",
    ]
    df_antennas = pd.read_csv(file_path_antennas, names=column_names, skiprows=1)

    df_5g_unfiltered = df_antennas[df_antennas["radio"].str.contains("LTE", na=False)]
    df_5g = df_5g_unfiltered[
        (df_5g_unfiltered["mcc"] == 460)
        & (df_5g_unfiltered["lat"].between(MIN_LAT, MAX_LAT))
        & (df_5g_unfiltered["lon"].between(MIN_LON, MAX_LON))
    ].copy()

    antenna_tree = cKDTree(df_5g[["lat", "lon"]].values)
    _, nearest_antennas = antenna_tree.query(df_taxi[["lat", "lon"]].values)

    taxi_count_per_antenna = np.bincount(nearest_antennas, minlength=len(df_5g))
    df_5g["taxi_count"] = taxi_count_per_antenna
    df_5g = df_5g[df_5g["taxi_count"] > 0]

    print("Number of antennas: ", len(df_5g))

    dist_matrix = cdist(df_5g[["lat", "lon"]].values, df_5g[["lat", "lon"]].values, metric='euclidean')

    # Delta = 1 - np.exp(-0.5 * dist_matrix)
    Delta = dist_matrix

    importance_values = df_5g["taxi_count"].values
    importance_values = 1 * (importance_values / importance_values.max())  # Normalize to [0, 1]
    df_5g["importance"] = importance_values

    return Delta, len(df_5g), df_5g, df_taxi, importance_values

def create_reduced_dataset(N_CLUSTERS):
    Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset()

    coords = df_5g[["lat", "lon"]].to_numpy()
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(coords)
    centroids = kmeans.cluster_centers_

    selected_antennas = []
    for center in centroids:
        closest_idx = np.argmin(
            [
                euclidean(center, [lat, lon])
                for lat, lon in zip(df_5g["lat"], df_5g["lon"])
            ]
        )
        selected_antennas.append(df_5g.iloc[closest_idx])

    df_selected = pd.DataFrame(selected_antennas)

    dist_matrix_reduced = cdist(df_selected[["lat", "lon"]].values, df_selected[["lat", "lon"]].values, metric='euclidean')
    Delta_reduced = 1 - np.exp(-0.5 * dist_matrix_reduced)

    importance_values_reduced = df_selected["taxi_count"].values
    importance_values_reduced = 2 * (importance_values_reduced / importance_values_reduced.max())
    df_selected["importance"] = importance_values_reduced

    return Delta_reduced, len(df_selected), df_selected, df_taxi, importance_values_reduced

if __name__ == "__main__":
    N_CLUSTERS = 25
    Delta_reduced, n_reduced, df_selected, df_taxi, importance_values_reduced = create_reduced_dataset(N_CLUSTERS)
    print("Reduced dataset created with", n_reduced, "antennas.")