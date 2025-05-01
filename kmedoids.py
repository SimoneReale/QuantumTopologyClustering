import dimod
from dimod import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler
import numpy as np
from visualization.plot_medoids_on_map import plot_medoids_on_map
from visualization.draw_charts import draw_chart_obj_fun_medoids
from sys import stdout
from scipy.spatial.distance import cdist
from create_medoids_cqm import create_cqm, create_native_bqm
from create_greedy_models import (
    create_cqm_greedy_kmedoids,
    create_bqm_greedy_kmedoids,
    create_bqm_max_min_distance,
    create_bqm_spread_out,
)
from create_greedy_models import create_bqm_min_distance_penalty
from create_new_models import create_cqm_even_spread, create_bqm_only_penalty
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from dwave.preprocessing.composites import FixVariablesComposite
from create_dataset import create_pechino_dataset, create_reduced_dataset
from solvers import (
    simple_simulated_annealing,
    greedy_kmedoids,
    quantum_annealing,
    hybrid_cqm_solve,
    p_median_kmedoids
)
from models_newest import create_bqm_even_spread, create_bqm_p_median


# n = 300  # Number of points
# np.random.seed(0)
# points = np.random.rand(n, 2) * 10  # Scale for better spacing
# dist_matrix = cdist(points, points, metric='euclidean')
# Delta = 1 - np.exp(-0.5 * dist_matrix)


# - k: Number of medoids
k = 20

Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(
    filter_radius=3000, is_importance_radius=True
)

# Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(N_CLUSTERS=100, filter_radius=3000)


# - alpha: Tradeoff parameter for dispersion
alpha = 1 * (1.0 / k)
# - beta: Tradeoff parameter for centrality
beta = 1 * (1.0 / n)


with alive_bar(1) as bar:
    print("Creo CQM")
    cqm, compute_objective = create_cqm(n, k, alpha, beta, Delta, importance_values)
    # cqm = create_kmeans_cqm(n, k, Delta)
    bar()


# bqm, compute_objective_new =  create_bqm_only_penalty(n, k, Delta, 3000, importance_values, alpha=2, beta = 1, lambda_=100)

bqm, compute_objective_new = create_bqm_even_spread(
    n,
    k,
    Delta,
    3000,
    c_p=15.0,
    c_s=1,
    lambda_=3.0,
    lagrange_multiplier=0.4,
    importance_values=importance_values,
)


# cqm = create_cqm_even_spread(n, k, Delta, 3000, 20000, 1)

print(f"bqm interactions and variables: {bqm.num_interactions} {bqm.num_variables}")

selected_medoids_greedy_bias = greedy_kmedoids(
    n, k, Delta, importance_values, bias=False
)

selected_medoids_p_median = p_median_kmedoids(
    df_taxi, df_5g, k, importance_values, demand_assign=1.0, selection_constr=1.0)


selected_medoids_bqm = simple_simulated_annealing(bqm, n, 5)

# print(f"Selected medoids bqm: {selected_medoids_bqm}")
# print(
#     f"Compute objective: {compute_objective_new(selected_medoids_bqm[0], Delta, importance_values)}"
# )

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
    [selected_medoids_p_median],
    compute_objective,
    Delta,
    df_taxi,
    df_5g,
    4000,
    figure_title="P median",
)

draw_chart_obj_fun_medoids(
    [selected_medoids_greedy_bias],
    compute_objective,
    Delta,
    df_taxi,
    df_5g,
    4000,
    figure_title="Greedy bias",
)


plot_medoids_on_map(
    df_5g,
    df_taxi,
    selected_medoids_bqm[0],
    Delta,
    4000,
    "New model shortest",
    plot_radius=True,
)

plot_medoids_on_map(
    df_5g,
    df_taxi,
    selected_medoids_greedy_bias,
    Delta,
    4000,
    "Greedy_medoids_bias",
    plot_radius=True,
)

plot_medoids_on_map(df_5g, df_taxi, selected_medoids_p_median, Delta, 4000, "P_median", plot_radius=True)

plt.show()
