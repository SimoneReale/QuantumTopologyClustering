from create_dataset import create_pechino_dataset, create_reduced_dataset
from clustering_quality import calculate_average_distances, plot_boxplot
from coverage_metrics import calculate_coverage, plot_coverage_boxplot
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
    # Example usage

    k = 20

    filter_radius = 3000

    NUM_CLUSTERS = 80

    # Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(
    #     filter_radius=3000, is_importance_radius=True
    # )

    Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(NUM_CLUSTERS, 3000, True, taxi_data_file="taxi_data/taxi_data_9.txt")
    


    # - alpha: Tradeoff parameter for dispersion
    alpha = 1 * (1.0 / k)
    # - beta: Tradeoff parameter for centrality
    beta = 1 * (1.0 / n)


    bqm, compute_objective_new = create_bqm_even_spread(
        n,
        k,
        Delta,
        3000,
        c_p=10.0,
        c_s=1,
        lambda_=3.0,
        lagrange_multiplier=1.5,
        importance_values=importance_values,
    )

    with alive_bar(1) as bar:
        print("Creo CQM")
        cqm, compute_objective = create_cqm(n, k, alpha, beta, Delta, importance_values)
        # cqm = create_kmeans_cqm(n, k, Delta)
        bar()


    # cqm = create_cqm_even_spread(n, k, Delta, 3000, 20000, 1)

    print(f"bqm interactions and variables: {bqm.num_interactions} {bqm.num_variables}")



    # selected_medoids_bqm = quantum_annealing(bqm, n, 5)

    selected_medoids_bqm = simple_simulated_annealing(bqm, n, 5)


    selected_medoids_p_median = [p_median_kmedoids(
        df_taxi, df_5g, k, importance_values, demand_assign=1.0, selection_constr=1.0, lambda_=0.1) for _ in range(1)]


    dict_medoids = {"Bqm": selected_medoids_bqm, "Pmedian": selected_medoids_p_median}


    avg_of_avgs_dict, all_distances_dict = calculate_average_distances(
        df_taxi, df_5g, dict_medoids
    )
    for key, avg in avg_of_avgs_dict.items():
        print(f"Average of the averages of the distances for {key}: {avg}")
    plot_boxplot(all_distances_dict)

    coverage_dict = calculate_coverage(df_taxi, df_5g, dict_medoids, filter_radius)
    for key, coverage in coverage_dict.items():
        print(f"Coverage percentages for {key}: {coverage}")
    plot_coverage_boxplot(coverage_dict)

    draw_chart_obj_fun_medoids(
        selected_medoids_bqm,
        compute_objective,
        Delta,
        df_taxi,
        df_5g,
        4000,
        figure_title="Simple bqm",
    )

    draw_chart_obj_fun_medoids(
        selected_medoids_p_median,
        compute_objective,
        Delta,
        df_taxi,
        df_5g,
        4000,
        figure_title="P_median",
    )


    plot_medoids_on_map(df_5g, df_taxi, selected_medoids_p_median[0], Delta, 3000, "P_median", plot_radius=True)

    plot_medoids_on_map(df_5g, df_taxi, selected_medoids_bqm[0], Delta, 3000, "Bqm", plot_radius=True)

   
    max_demand_per_medoid, demand_to_medoid_ratio_variance, avg = calculate_fairness_and_equity_metrics(df_taxi, df_5g, selected_medoids_bqm[0], 3000, assign_type="radius")
    print(f"Maximum Demand per Medoid bqm: {max_demand_per_medoid}")
    print(f"Demand-to-Medoid Ratio std: {demand_to_medoid_ratio_variance}")
    print("Avergae demand per medoid: ", avg)

    max_demand_per_medoid, demand_to_medoid_ratio_variance, avg = calculate_fairness_and_equity_metrics(df_taxi, df_5g, selected_medoids_p_median[0], 3000, assign_type="radius")
    print(f"Maximum Demand per Medoid p-median: {max_demand_per_medoid}")
    print(f"Demand-to-Medoid Ratio std: {demand_to_medoid_ratio_variance}")
    print("Avergae demand per medoid: ", avg)


    m, area_km2 = plot_convex_hull(df_5g, dict_medoids["Bqm"][0])
    print(f"Area of the convex hull bqm: {area_km2} km^2 and density: {k / area_km2}")
    m.save("folium_output/convex_hull_map_bqm.html")

    m, area_km2 = plot_convex_hull(df_5g, dict_medoids["Pmedian"][0])
    print(f"Area of the convex hull Pmedian: {area_km2} km^2 and density: {k / area_km2}")
    m.save("folium_output/convex_hull_map_Pmedian.html")


    plt.show()
