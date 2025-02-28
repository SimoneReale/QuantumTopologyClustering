import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

def create_pechino_dataset():
    MIN_LAT, MAX_LAT = 39.5, 41.0
    MIN_LON, MAX_LON = 115.5, 117.5

    df_taxi = pd.read_csv(
        "taxi_data/taxi_data_7.txt",
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

    dist_matrix = cdist(df_5g[["lat", "lon"]].values, df_5g[["lat", "lon"]].values, metric='euclidean')

    Delta = 1 - np.exp(-0.5 * dist_matrix)

    importance_values = df_5g["taxi_count"].values
    importance_values = 1.5 * (importance_values / importance_values.max())  # Normalize to [0, 1]
    

    

    return Delta, len(df_5g), df_5g, df_taxi, importance_values


if __name__ == "__main__":
    create_pechino_dataset()