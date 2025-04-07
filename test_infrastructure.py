import os
import json
from create_dataset import create_pechino_dataset, create_reduced_dataset
from clustering_quality import calculate_average_distances, calculate_min_max_ratio
from coverage_metrics import calculate_coverage
from spatial_distribution_metrics import calculate_medoid_density, plot_convex_hull
import matplotlib.pyplot as plt
from models_newest import (
    create_bqm_even_spread,
    calculate_lambda,
    choose_weights,
    create_cqm_even_spread,
)
from create_medoids_cqm import create_cqm
from solvers import (
    simple_simulated_annealing,
    p_median_kmedoids,
    quantum_annealing,
    sklearn_kmedoids,
    hybrid_cqm_solve,
)
from visualization.draw_charts import draw_chart_obj_fun_medoids
from alive_progress import alive_bar
from visualization.plot_medoids_on_map import plot_medoids_on_map
from demand_balance_metrics import calculate_fairness_and_equity_metrics
from create_medoids_cqm import create_native_bqm
import time
import numpy as np

solvers = ["bqm_even_spread_simulated_annealing", "p_median_classic"]


def test_pmedian(
    k,
    taxi_data_file="taxi_data/taxi_data_1.txt",
    filter_radius=10000,
    NUM_CLUSTERS_REDUCED=1000,
    gamma=1.0,
):
    Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(
        NUM_CLUSTERS_REDUCED,
        filter_radius=filter_radius,
        is_importance_radius=True,
        taxi_data_file=taxi_data_file,
    )
    selected_medoids_p_median = p_median_kmedoids(
        df_taxi, df_5g, k, importance_values, gamma=gamma
    )
    return Delta, n, df_5g, df_taxi, importance_values, selected_medoids_p_median


def test_bqm_even_spread_simulated_annealing(
    k,
    taxi_data_file="taxi_data/taxi_data_1.txt",
    filter_radius=10000,
    NUM_CLUSTERS_REDUCED=1000,
    c_p=10.0,
    c_s=1,
    lambda_=3.0,
    lagrange_multiplier=1.5,
    radius_model=6000,
):
    Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(
        NUM_CLUSTERS_REDUCED,
        filter_radius=filter_radius,
        is_importance_radius=True,
        taxi_data_file=taxi_data_file,
    )
    bqm, _ = create_bqm_even_spread(
        n,
        k,
        Delta,
        radius_model,
        c_p=c_p,
        c_s=c_s,
        lambda_=lambda_,
        lagrange_multiplier=lagrange_multiplier,
        importance_values=importance_values,
    )
    selected_medoids_bqm = simple_simulated_annealing(bqm, n, 5)
    return Delta, n, df_5g, df_taxi, importance_values, selected_medoids_bqm[0]


def test_cqm_even_spread_hybrid(
    k,
    taxi_data_file="taxi_data/taxi_data_1.txt",
    filter_radius=10000,
    NUM_CLUSTERS_REDUCED=1000,
    c_p=10.0,
    c_s=1,
    lambda_=3.0,
    lagrange_multiplier=1.5,
    radius_model=3000,
):
    Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(
        NUM_CLUSTERS_REDUCED,
        filter_radius=filter_radius,
        is_importance_radius=True,
        taxi_data_file=taxi_data_file,
    )
    cqm, _ = create_cqm_even_spread(
        n,
        k,
        Delta,
        radius_model,
        lagrange_multiplier,
        importance_values,
        c_p=c_p,
        c_s=c_s,
        lambda_=lambda_,
    )
    selected_medoids_cqm = hybrid_cqm_solve(cqm, n, 5)
    return Delta, n, df_5g, df_taxi, importance_values, selected_medoids_cqm[0]


def test_cqm_kmedoids_hybrid(k,
    taxi_data_file="taxi_data/taxi_data_1.txt",
    filter_radius=10000,
    NUM_CLUSTERS_REDUCED=1000,
    lambda_=1):
    Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(
        NUM_CLUSTERS_REDUCED,
        filter_radius=filter_radius,
        is_importance_radius=True,
        taxi_data_file=taxi_data_file,
    )

    alpha = 1 / k
    beta = 1 / n
    cqm, _ = create_cqm(n, k, alpha, beta, Delta, importance_values, lambda_=lambda_)
    selected_medoids_cqm = hybrid_cqm_solve(cqm, n, 5)
    return Delta, n, df_5g, df_taxi, importance_values, selected_medoids_cqm[0]


def test_bqm_even_spread_REAL_annealing(
    k,
    taxi_data_file="taxi_data/taxi_data_1.txt",
    filter_radius=10000,
    NUM_CLUSTERS_REDUCED=1000,
    c_p=10.0,
    c_s=1,
    lambda_=3.0,
    lagrange_multiplier=1.5,
):
    Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(
        NUM_CLUSTERS_REDUCED,
        filter_radius=filter_radius,
        is_importance_radius=True,
        taxi_data_file=taxi_data_file,
    )

    bqm, _ = create_bqm_even_spread(
        n,
        k,
        Delta,
        filter_radius,
        c_p=c_p,
        c_s=c_s,
        lambda_=lambda_,
        lagrange_multiplier=lagrange_multiplier,
        importance_values=importance_values,
    )
    selected_medoids_bqm = quantum_annealing(bqm, n, 5)
    return Delta, n, df_5g, df_taxi, importance_values, selected_medoids_bqm[0]


def test_bqm_kmedoid_native(
    k,
    taxi_data_file="taxi_data/taxi_data_1.txt",
    filter_radius=10000,
    NUM_CLUSTERS_REDUCED=1000,
):
    Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(
        NUM_CLUSTERS_REDUCED,
        filter_radius=filter_radius,
        is_importance_radius=True,
        taxi_data_file=taxi_data_file,
    )
    alpha = 1 / k
    beta = 1 / n
    lagrange_multiplier = 2
    bqm = create_native_bqm(n, k, alpha, beta, Delta, lagrange_multiplier)
    selected_medoids_bqm = simple_simulated_annealing(bqm, n, 5)
    print(f"Selected medoids {selected_medoids_bqm}")
    return Delta, n, df_5g, df_taxi, importance_values, selected_medoids_bqm[0]


def test_bqm_kmedoid_native_fully_quantum(
    k,
    taxi_data_file="taxi_data/taxi_data_1.txt",
    filter_radius=10000,
    NUM_CLUSTERS_REDUCED=1000,
):
    Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(
        NUM_CLUSTERS_REDUCED,
        filter_radius=filter_radius,
        is_importance_radius=True,
        taxi_data_file=taxi_data_file,
    )
    alpha = 1 / k
    beta = 1 / n
    lagrange_multiplier = 2
    bqm = create_native_bqm(n, k, alpha, beta, Delta, lagrange_multiplier)
    selected_medoids_bqm = quantum_annealing(bqm, n, 5)
    print(f"Selected medoids {selected_medoids_bqm}")
    return Delta, n, df_5g, df_taxi, importance_values, selected_medoids_bqm[0]


def test_k_medoids_sklearn(
    k,
    taxi_data_file="taxi_data/taxi_data_1.txt",
    filter_radius=10000,
    NUM_CLUSTERS_REDUCED=1000,
):
    Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(
        NUM_CLUSTERS_REDUCED,
        filter_radius=filter_radius,
        is_importance_radius=True,
        taxi_data_file=taxi_data_file,
    )
    selected_medoids = sklearn_kmedoids(df_5g, k)
    return Delta, n, df_5g, df_taxi, importance_values, selected_medoids


if __name__ == "__main__":
    dict_of_methods = {
        0: "k_medoids_native_bqm",
        1: "p_median_importance",
        2: "k_medoids_sklearn",
        3: "bqm_even_spread_simulated_annealing",
        4: "bqm_even_spread_REAL_annealing",
        5: "cqm_even_spread_hybrid",
        6: "cqm_kmedoids_hybrid",
        7: "k_medoids_native_REAL_annealing"
    }

    radius = 3000
    k = 20
    NUM_CLUSTERS_REDUCED = 100

    method = dict_of_methods[6]

    if NUM_CLUSTERS_REDUCED < 1000:
        method = method + "_reduced"

    all_solutions = []

    single_range = [1]
    full_range = range(1, 11)

    list_methods = ["k_medoids_native_bqm"]

    execution_times = []  # To store execution times for each split
    for i in range(5):
        for method in list_methods:
            for i in single_range:

                filename = f"splits/split_{i}.txt"

                print(
                    f"\nSTO GESTENDO {filename} con metodo {method} ##################################################\n"
                )

                start_time = time.time()  # Start timing

                if "k_medoids_native_bqm" in method:
                    Delta, n, df_5g, df_taxi, importance_values, selected_medoids = (
                        test_bqm_kmedoid_native(
                            k,
                            filename,
                            filter_radius=radius,
                            NUM_CLUSTERS_REDUCED=NUM_CLUSTERS_REDUCED,
                        )
                    )
                elif "p_median_importance" in method:
                    Delta, n, df_5g, df_taxi, importance_values, selected_medoids = (
                        test_pmedian(
                            k,
                            filename,
                            filter_radius=radius,
                            NUM_CLUSTERS_REDUCED=NUM_CLUSTERS_REDUCED,
                            gamma=0.2,
                        )
                    )

                elif "k_medoids_native_REAL_annealing" in method:
                    Delta, n, df_5g, df_taxi, importance_values, selected_medoids = (
                        test_bqm_kmedoid_native_fully_quantum(
                            k,
                            filename,
                            filter_radius=radius,
                            NUM_CLUSTERS_REDUCED=NUM_CLUSTERS_REDUCED,
                        )
                    )
                
                elif "cqm_kmedoids_hybrid" in method:
                    Delta, n, df_5g, df_taxi, importance_values, selected_medoids = (
                        test_cqm_kmedoids_hybrid(
                            k,
                            filename,
                            filter_radius=radius,
                            NUM_CLUSTERS_REDUCED=NUM_CLUSTERS_REDUCED,
                            lambda_= 0.005
                        )
                    )
                
                elif "k_medoids_sklearn" in method:
                    Delta, n, df_5g, df_taxi, importance_values, selected_medoids = (
                        test_k_medoids_sklearn(
                            k,
                            filename,
                            filter_radius=radius,
                            NUM_CLUSTERS_REDUCED=NUM_CLUSTERS_REDUCED,
                        )
                    )
                elif "bqm_even_spread_simulated_annealing" in method:
                    Delta, n, df_5g, df_taxi, importance_values, selected_medoids = (
                        test_bqm_even_spread_simulated_annealing(
                            k,
                            filename,
                            filter_radius=radius,
                            NUM_CLUSTERS_REDUCED=NUM_CLUSTERS_REDUCED,
                            c_p=10.0,
                            c_s=1,
                            lambda_=3.0,
                            lagrange_multiplier=1.0,
                            radius_model=4000,
                        )
                    )
                elif "bqm_even_spread_REAL_annealing" in method:
                    Delta, n, df_5g, df_taxi, importance_values, selected_medoids = (
                        test_bqm_even_spread_REAL_annealing(
                            k,
                            filename,
                            filter_radius=radius,
                            NUM_CLUSTERS_REDUCED=NUM_CLUSTERS_REDUCED,
                            c_p=10.0,
                            c_s=1,
                            lambda_=1.0,
                            lagrange_multiplier=2.5,
                        )
                    )
                elif "cqm_even_spread_hybrid" in method:
                    Delta, n, df_5g, df_taxi, importance_values, selected_medoids = (
                        test_cqm_even_spread_hybrid(
                            k,
                            filename,
                            filter_radius=radius,
                            NUM_CLUSTERS_REDUCED=NUM_CLUSTERS_REDUCED,
                            c_p=10.0,
                            c_s=1,
                            lambda_=3.0,
                            lagrange_multiplier=1.0,
                            radius_model=4000,
                        )
                    )
                else:
                    Delta, n, df_5g, df_taxi, importance_values, selected_medoids = (
                        test_pmedian(
                            k,
                            filename,
                            filter_radius=radius,
                            NUM_CLUSTERS_REDUCED=NUM_CLUSTERS_REDUCED,
                            gamma=0.2,
                        )
                    )
                    print("METHOD ERROR, let's try with p-median importance!")

                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)  # Store the execution time

                print(f"Execution time for {filename} with method {method}: {execution_time:.2f} seconds")

                # Clustering Quality
                max_distance, min_distance, avg_distance, std_distance, all_distances = (
                    calculate_average_distances(df_taxi, df_5g, selected_medoids)
                )
                min_max_ratio = calculate_min_max_ratio(df_taxi, df_5g, selected_medoids)

                # Coverage
                coverage_percentage = calculate_coverage(
                    df_taxi, df_5g, selected_medoids, radius
                )

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
                    df_5g,
                    selected_medoids,
                    filename=f"{output_dir}/hull_{method}_dataset_{i}.html",
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

                single_dataset_sol = {
                    "file_name": filename,
                    "method": method,
                    "Delta": Delta.tolist(),
                    "number_of_antennas": n,
                    "df_5g": df_5g.to_dict(),
                    "df_taxi": df_taxi.to_dict(),
                    "importance_values": importance_values.tolist(),
                    "selected_medoids": selected_medoids,
                    "num_of_selected_medoids": len(selected_medoids),
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

                all_solutions.append(single_dataset_sol)

            with open(f"{method}/all_solutions_{method}.json", "w") as f:
                json.dump(all_solutions, f, indent=4)

    # Calculate and print average and standard deviation of execution times
    avg_time = np.mean(execution_times)
    std_time = np.std(execution_times)
    print(f"\nAverage execution time: {avg_time:.2f} seconds")
    print(f"Standard deviation of execution times: {std_time:.2f} seconds")
