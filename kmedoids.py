import dimod
from dimod import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler
import numpy as np
from visualization import visualize_graph, draw_chart_obj_fun_medoids, plot_medoids_on_map
from sys import stdout
from scipy.spatial.distance import cdist
from create_medoids_cqm import create_cqm, create_bqm, create_native_bqm, create_kmeans_cqm
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from dwave.preprocessing.composites import FixVariablesComposite
from create_dataset import create_pechino_dataset, create_reduced_dataset
from solvers import simple_simulated_annealing, roof_duality_simulated_annealing, exact_cqm_solve




# n = 300  # Number of points
# np.random.seed(0)
# points = np.random.rand(n, 2) * 10  # Scale for better spacing
# dist_matrix = cdist(points, points, metric='euclidean')
# Delta = 1 - np.exp(-0.5 * dist_matrix)


# - k: Number of medoids
k = 5   

Delta, n, df_5g, df_taxi, importance_values = create_reduced_dataset(N_CLUSTERS=20)

#Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset()

# - alpha: Tradeoff parameter for dispersion
alpha = 1 * (1.0 / k)
# - beta: Tradeoff parameter for centrality
beta = 1 * (1.0 / n)


with alive_bar(1) as bar:
    print("Creo CQM")
    #cqm, compute_objective = create_cqm(n, k, alpha, beta, Delta, importance_values)
    cqm = create_kmeans_cqm(n, k, Delta)
    bar()

with alive_bar(1) as bar:
    print("Creo BQM")
    bqm, invert = create_bqm(cqm)
    bar()


bqm = create_native_bqm(n, k, alpha, beta, Delta, importance_values)

selected_medoids_bqm = simple_simulated_annealing(bqm, n, 20)
selected_medoids_exact = exact_cqm_solve(cqm, n, 20)



# visualize_graph(n, Delta, points, selected_medoids_bqm, alpha, beta, compute_objective, "Simple bqm")
# visualize_graph(n, Delta, points, selected_medoids_bqm_fixed, alpha, beta, compute_objective, "Fixed roof duality")

# draw_chart_obj_fun_medoids(selected_medoids_bqm, compute_objective, figure_title="Simple bqm")
# draw_chart_obj_fun_medoids(selected_medoids_exact, compute_objective, figure_title="Exact cqm")
# draw_chart_obj_fun(selected_medoids_bqm_fixed, compute_objective, figure_title="Fixed bqm")


plot_medoids_on_map(df_5g, df_taxi, selected_medoids_bqm[0], "Simple_bqm_medoids")
plot_medoids_on_map(df_5g, df_taxi, selected_medoids_exact[0], "Exact_cqm_medoids")
# plot_medoids_on_map(df_5g, df_taxi, selected_medoids_bqm_fixed[0], "Fixed_bqm_medoids")

plt.show()
