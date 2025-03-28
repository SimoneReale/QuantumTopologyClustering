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
import seaborn as sns


def plot_boxplots(all_solutions, output_dir):
    metrics = [
        "max_distance",
        "avg_distance",
        "std_distance",
        "min_max_ratio",
        "coverage_percentage",
        "max_demand_per_medoid",
        "demand_to_medoid_ratio_std",
        "avg_demand_per_medoid",
        "area_km2",
        "medoid_density"
    ]

    for metric in metrics:
        data = [sol[metric] for sol in all_solutions]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data)
        plt.title(f'Box Plot of {metric}')
        plt.xlabel('Solutions')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.savefig(f"{output_dir}/boxplot_{metric}.png")
        plt.close()

if __name__ == "__main__":
    method = "p_median"
    json_file = f"{method}/all_solutions_{method}.json"

    with open(json_file, "r") as f:
        all_solutions = json.load(f)

    output_dir = f"{method}/boxplots"
    os.makedirs(output_dir, exist_ok=True)

    plot_boxplots(all_solutions, output_dir)