import numpy as np
import dimod
from dwave.system import LeapHybridCQMSampler, LeapHybridSampler
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel
from dimod import quicksum
from create_dataset import create_pechino_dataset
from config import tokenuccio
import os
import json
from create_dataset import create_pechino_dataset, create_reduced_dataset
from clustering_quality import calculate_average_distances, calculate_min_max_ratio
from coverage_metrics import calculate_coverage
from spatial_distribution_metrics import calculate_medoid_density, plot_convex_hull
import matplotlib.pyplot as plt
from models_newest import create_bqm_even_spread
from create_medoids_cqm import create_cqm
from solvers import simple_simulated_annealing, p_median_kmedoids, quantum_annealing
from visualization.draw_charts import draw_chart_obj_fun_medoids
from alive_progress import alive_bar
from visualization.plot_medoids_on_map import plot_medoids_on_map
from demand_balance_metrics import calculate_fairness_and_equity_metrics


def p_median_cqm(df_taxi, df_5g, k, importance_values, gamma=1.0):
    """
    Solves the k-medoids problem using a Constrained Quadratic Model (CQM) for D-Wave.

    Parameters:
    - df_taxi: DataFrame containing the taxi data (demand points)
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - k: Number of medoids
    - importance_values: Array of importance values for each point
    - gamma: Exponent controlling the influence of importance

    Returns:
    - selected_medoids: List of selected medoids
    """
    n = len(df_5g)
    m = len(df_taxi)

    # Calculate the Delta matrix (distance matrix between demand points and service points)
    Delta = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            Delta[i, j] = np.sqrt(
                (df_5g.iloc[i]["lat"] - df_taxi.iloc[j]["lat"]) ** 2
                + (df_5g.iloc[i]["lon"] - df_taxi.iloc[j]["lon"]) ** 2
            )

    # Create the modified distance matrix incorporating importance values
    modified_Delta = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            modified_Delta[i, j] = Delta[i, j] / (importance_values[i] ** gamma)

    # Create the CQM problem
    cqm = ConstrainedQuadraticModel()

    # Decision variables
    z = {i: dimod.Binary(f"z_{i}") for i in range(n)}
    x = {(i, j): dimod.Binary(f"x_{i}_{j}") for i in range(n) for j in range(m)}

    # Objective function: Minimize total assignment cost
    cqm.set_objective(
        quicksum(modified_Delta[i, j] * x[i, j] for i in range(n) for j in range(m))
    )

    # Constraint: Each demand point must be assigned to exactly one service location
    for j in range(m):
        cqm.add_constraint(
            quicksum(x[i, j] for i in range(n)) == 1, label=f"demand_assignment_{j}"
        )

    # Constraint: A demand point can only be assigned to an active service location
    for i in range(n):
        for j in range(m):
            cqm.add_constraint(x[i, j] - z[i] <= 0, label=f"assignment_active_{i}_{j}")

    # Constraint: Select exactly k medoids
    cqm.add_constraint(quicksum(z[i] for i in range(n)) == k, label="select_k_medoids")

    return cqm


def p_median_bqm(df_taxi, df_5g, k, importance_values, gamma=1.0):
    """
    Solves the k-medoids problem using a Binary Quadratic Model (BQM) for D-Wave.

    Parameters:
    - df_taxi: DataFrame containing the taxi data (demand points)
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - k: Number of medoids
    - importance_values: Array of importance values for each point
    - gamma: Exponent controlling the influence of importance

    Returns:
    - selected_medoids: List of selected medoids
    """
    n = len(df_5g)
    m = len(df_taxi)

    # Calculate the Delta matrix (distance matrix between demand points and service points)
    Delta = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            Delta[i, j] = np.sqrt(
                (df_5g.iloc[i]["lat"] - df_taxi.iloc[j]["lat"]) ** 2
                + (df_5g.iloc[i]["lon"] - df_taxi.iloc[j]["lon"]) ** 2
            )

    # Create the modified distance matrix incorporating importance values
    modified_Delta = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            modified_Delta[i, j] = Delta[i, j] / (importance_values[i] ** gamma)

    # Create the BQM problem
    bqm = BinaryQuadraticModel("BINARY")

    # Decision variables
    z = {i: dimod.Binary(f"z_{i}") for i in range(n)}
    x = {(i, j): dimod.Binary(f"x_{i}_{j}") for i in range(n) for j in range(m)}

    # Objective function: Minimize total assignment cost
    for i in range(n):
        for j in range(m):
            bqm.add_variable(f"x_{i}_{j}", modified_Delta[i, j])

    # Constraint: Each demand point must be assigned to exactly one service location
    for j in range(m):
        bqm.add_linear_equality_constraint(
            [(f"x_{i}_{j}", 1) for i in range(n)], constant=-1, lagrange_multiplier=1.0
        )

    # Constraint: A demand point can only be assigned to an active service location
    for i in range(n):
        for j in range(m):
            bqm.add_linear_inequality_constraint(
                [(f"x_{i}_{j}", 1), (f"z_{i}", -1)],
                constant=0,
                lagrange_multiplier=1.0,
                label=f"assignment_active_{i}_{j}",
            )

    # Constraint: Select exactly k medoids
    bqm.add_linear_equality_constraint(
        [(f"z_{i}", 1) for i in range(n)], constant=-k, lagrange_multiplier=1.0
    )

    return bqm


if __name__ == "__main__":
    method = "p_median_cqm_hybrid"
    radius = 3000
    i = 8
    k = 20

    Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(
        filter_radius=radius,
        is_importance_radius=True,
        taxi_data_file=f"taxi_data/taxi_data_{i}.txt",
    )

    # bqm = p_median_bqm(df_taxi, df_5g, 20, importance_values, gamma=1.0)

    cqm = p_median_cqm(df_taxi, df_5g, k, importance_values, gamma=0.2)

    print(
        f"cqm quadratic variables {cqm.num_quadratic_variables()} num of biases {cqm.num_biases()}"
    )

    sampler = LeapHybridCQMSampler(token=tokenuccio)
    sampleset = sampler.sample_cqm(cqm)
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
    best_sample = feasible_sampleset.first.sample

    selected_medoids = [i for i in range(n) if best_sample[f"z_{i}"] == 1]

    print("selected medoids: ", selected_medoids)

    # Clustering Quality
    max_distance, avg_distance, std_distance, all_distances = (
        calculate_average_distances(df_taxi, df_5g, selected_medoids)
    )
    min_max_ratio = calculate_min_max_ratio(df_taxi, df_5g, selected_medoids)

    # Coverage
    coverage_percentage = calculate_coverage(df_taxi, df_5g, selected_medoids, radius)

    # Demand
    max_demand_per_medoid, demand_to_medoid_ratio_std, avg_demand_per_medoid = (
        calculate_fairness_and_equity_metrics(
            df_taxi, df_5g, selected_medoids, radius, assign_type="radius"
        )
    )

    # Spatial distribution
    output_dir = f"{method}/{i}"
    os.makedirs(output_dir, exist_ok=True)
    area_km2 = plot_convex_hull(
        df_5g, selected_medoids, filename=f"{output_dir}/hull_{method}_dataset_{i}.html"
    )
    medoid_density = calculate_medoid_density(selected_medoids, area_km2)

    # Plot medoids
    plot_medoids_on_map(
        df_5g,
        df_taxi,
        selected_medoids,
        Delta,
        radius,
        f"{method} data {i}",
        plot_radius=True,
        plot_distances=False,
        filename=f"{output_dir}/plot_{method}_dataset_{i}.html",
    )

    # Store all metrics in the dictionary
    single_dataset_sol = {
        "file_name": f"taxi_data/taxi_data_{i}.txt",
        "Delta": Delta.tolist(),
        "n": n,
        "df_5g": df_5g.to_dict(),
        "df_taxi": df_taxi.to_dict(),
        "importance_values": importance_values.tolist(),
        "selected_medoids": selected_medoids,
        "max_distance": max_distance,
        "avg_distance": avg_distance,
        "std_distance": std_distance,
        "all_distances": all_distances,
        "min_max_ratio": min_max_ratio,
        "coverage_percentage": coverage_percentage,
        "max_demand_per_medoid": max_demand_per_medoid,
        "demand_to_medoid_ratio_std": demand_to_medoid_ratio_std,
        "avg_demand_per_medoid": avg_demand_per_medoid,
        "area_km2": area_km2,
        "medoid_density": medoid_density,
    }

    with open(f"{output_dir}/single_dataset_{method}_sol_{i}.json", "w") as f:
        json.dump(single_dataset_sol, f, indent=4)

    # print(f"bqm interactions and variables: {bqm.num_interactions} {bqm.num_variables}")

