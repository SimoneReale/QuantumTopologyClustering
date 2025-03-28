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



if __name__ == "__main__":
    k = 20

    filter_radius = 3000

    Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(
        filter_radius=3000, is_importance_radius=True, taxi_data_file="taxi_data/taxi_data_1.txt"
    )

    # - alpha: Tradeoff parameter for dispersion
    alpha = 1 * (1.0 / k)
    # - beta: Tradeoff parameter for centrality
    beta = 1 * (1.0 / n)


    with alive_bar(1) as bar:
        print("Creo CQM")
        cqm, compute_objective = create_cqm(n, k, alpha, beta, Delta, importance_values)
        # cqm = create_kmeans_cqm(n, k, Delta)
        bar()


    # cqm = create_cqm_even_spread(n, k, Delta, 3000, 20000, 1)

    method = "pmedian_importance_new"
    i = 1
    radius = 3000

    selected_medoids = p_median_kmedoids(df_taxi, df_5g, k, importance_values, demand_assign=1.0, selection_constr=1.0, gamma=0.2, lagrange_multiplier=1.0)

    # Clustering Quality
    max_distance, avg_distance, std_distance, all_distances = calculate_average_distances(df_taxi, df_5g, selected_medoids)
    min_max_ratio = calculate_min_max_ratio(df_taxi, df_5g, selected_medoids)

    # Coverage
    coverage_percentage = calculate_coverage(df_taxi, df_5g, selected_medoids, radius)

    # Demand
    max_demand_per_medoid, demand_to_medoid_ratio_std, avg_demand_per_medoid = calculate_fairness_and_equity_metrics(df_taxi, df_5g, selected_medoids, radius, assign_type="radius")

    # Spatial distribution
    output_dir = f"{method}/{i}"
    os.makedirs(output_dir, exist_ok=True)
    area_km2 = plot_convex_hull(df_5g, selected_medoids, filename = f"{output_dir}/hull_{method}_dataset_{i}.html")
    medoid_density = calculate_medoid_density(selected_medoids, area_km2)

    # Plot medoids
    plot_medoids_on_map(df_5g, df_taxi, selected_medoids, Delta, radius, f"{method} data {i}", plot_radius=True, plot_distances=False, filename=f"{output_dir}/plot_{method}_dataset_{i}.html")

    draw_chart_obj_fun_medoids(
        [selected_medoids],
        compute_objective,
        Delta,
        df_taxi,
        df_5g,
        3000,
        figure_title="P_median",
    )

    plt.show()