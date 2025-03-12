import dimod
from dimod import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler
import numpy as np
from visualization import visualize_graph, draw_chart_obj_fun_medoids, plot_medoids_on_map
from sys import stdout
from scipy.spatial.distance import cdist
from create_medoids_cqm import create_cqm, create_native_bqm
from create_greedy_models import create_cqm_greedy_kmedoids, create_bqm_greedy_kmedoids, create_bqm_max_min_distance, create_bqm_spread_out
from create_greedy_models import create_bqm_min_distance_penalty
from create_new_models import create_cqm_even_spread, create_bqm_only_penalty
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from dwave.preprocessing.composites import FixVariablesComposite
from create_dataset import create_pechino_dataset, create_reduced_dataset
from solvers import simple_simulated_annealing, greedy_kmedoids, quantum_annealing, hybrid_cqm_solve
from models_newest import create_bqm_even_spread




# n = 300  # Number of points
# np.random.seed(0)
# points = np.random.rand(n, 2) * 10  # Scale for better spacing
# dist_matrix = cdist(points, points, metric='euclidean')
# Delta = 1 - np.exp(-0.5 * dist_matrix)


# - k: Number of medoids
k = 15   

Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(filter_radius=3000, is_importance_radius=True)

#Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(N_CLUSTERS=100, filter_radius=3000)


# - alpha: Tradeoff parameter for dispersion
alpha = 1 * (1.0 / k)
# - beta: Tradeoff parameter for centrality
beta = 1 * (1.0 / n)


with alive_bar(1) as bar:
    print("Creo CQM")
    cqm, compute_objective = create_cqm(n, k, alpha, beta, Delta, importance_values)
    #cqm = create_kmeans_cqm(n, k, Delta)
    bar()


#bqm, compute_objective_new =  create_bqm_only_penalty(n, k, Delta, 3000, importance_values, alpha=2, beta = 1, lambda_=100)

bqm, compute_objective_new =  create_bqm_even_spread(n, k, Delta, 3000, c_p=4.5, c_s=1, lambda_=1, lagrange_multiplier=0.4, importance_values=importance_values)


#cqm = create_cqm_even_spread(n, k, Delta, 3000, 20000, 1)

print(f"bqm interactions and variables: {bqm.num_interactions} {bqm.num_variables}")

selected_medoids_greedy_bias = greedy_kmedoids(n, k, Delta, importance_values, bias=False)

#selected_medoids_greedy_no_bias = greedy_kmedoids(n, k, Delta, importance_values, bias=False)
selected_medoids_bqm = simple_simulated_annealing(bqm, n, 5)
#selected_medoids_cqm = hybrid_cqm_solve(cqm, n, 5)
print(f"Selected medoids bqm: {selected_medoids_bqm}")
print(f"Compute objective: {compute_objective_new(selected_medoids_bqm[0], Delta, importance_values)}")
#selected_medoids_ortools = ortools_cqm_solve(model=ortools_model, num_of_nodes=n)
#selected_medoids_exact = exact_cqm_solve(cqm, n, 20)



# visualize_graph(n, Delta, points, selected_medoids_bqm, alpha, beta, compute_objective, "Simple bqm")
# visualize_graph(n, Delta, points, selected_medoids_bqm_fixed, alpha, beta, compute_objective, "Fixed roof duality")

draw_chart_obj_fun_medoids(selected_medoids_bqm, compute_objective, Delta, figure_title="Simple bqm")
#draw_chart_obj_fun_medoids(selected_medoids_cqm, compute_objective, Delta, figure_title="Simple cqm")
# draw_chart_obj_fun_medoids(selected_medoids_exact, compute_objective, figure_title="Exact cqm")
# draw_chart_obj_fun(selected_medoids_bqm_fixed, compute_objective, figure_title="Fixed bqm")
#draw_chart_obj_fun_medoids([selected_medoids_ortools], compute_objective, figure_title="Ortools cqm")
draw_chart_obj_fun_medoids([selected_medoids_greedy_bias], compute_objective, Delta, figure_title="Greedy bias")
#draw_chart_obj_fun_medoids([selected_medoids_greedy_no_bias], compute_objective, figure_title="Greedy no bias")


plot_medoids_on_map(df_5g, df_taxi, selected_medoids_bqm[0], Delta, 4000, "New model shortest", plot_radius=False)
#plot_medoids_on_map(df_5g, df_taxi, selected_medoids_cqm[0], Delta, 4000, "New model cqm")
#plot_medoids_on_map(df_5g, df_taxi, selected_medoids_exact[0], "Exact_cqm_medoids")
#plot_medoids_on_map(df_5g, df_taxi, selected_medoids_ortools, "Ortools_cqm_medoids")
# plot_medoids_on_map(df_5g, df_taxi, selected_medoids_bqm_fixed[0], "Fixed_bqm_medoids")
plot_medoids_on_map(df_5g, df_taxi, selected_medoids_greedy_bias, Delta, 4000, "Greedy_medoids_bias", plot_radius=False)
#plot_medoids_on_map(df_5g, df_taxi, selected_medoids_greedy_no_bias, "Greedy_medoids_no_bias")

plt.show()
