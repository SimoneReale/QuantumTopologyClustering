import dimod
from dimod import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler
import numpy as np
from visualization import visualize_graph, draw_chart_obj_fun, plot_medoids_on_map
from sys import stdout
from scipy.spatial.distance import cdist
from create_medoids_cqm import create_cqm, create_bqm
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from dwave.preprocessing.composites import FixVariablesComposite
from create_dataset import create_pechino_dataset




# n = 300  # Number of points
# np.random.seed(0)
# points = np.random.rand(n, 2) * 10  # Scale for better spacing
# dist_matrix = cdist(points, points, metric='euclidean')
# Delta = 1 - np.exp(-0.5 * dist_matrix)



Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset()

# - k: Number of medoids
k = 20   
# - alpha: Tradeoff parameter for dispersion
alpha = 1 * (1.0 / k)
# - beta: Tradeoff parameter for centrality
beta = 1 * (1.0 / n)

with alive_bar(1) as bar:
    print("Creo CQM")
    cqm, compute_objective = create_cqm(n, k, alpha, beta, Delta, importance_values)
    bar()

with alive_bar(1) as bar:
    print("Creo BQM")
    create_bqm(cqm)
    bar()

bqm, invert = create_bqm(cqm)

with alive_bar(1) as bar:
    print("Fixed roof duality")
    sampler_fixed = FixVariablesComposite(SimulatedAnnealingSampler(), algorithm='roof_duality')
    solution_bqm_fixed = sampler_fixed.sample(bqm)
    selected_medoids_bqm_fixed = [[i for i in range(n) if sample[f'z_{i}'] == 1] for sample in solution_bqm_fixed.samples()][:6]
    bar()

with alive_bar(1) as bar:   
    print("No preprocessing")
    sampler = SimulatedAnnealingSampler()
    solution_bqm = sampler.sample(bqm)
    selected_medoids_bqm = [[i for i in range(n) if sample[f'z_{i}'] == 1] for sample in solution_bqm.samples()][:6]
    bar()

# visualize_graph(n, Delta, points, selected_medoids_bqm, alpha, beta, compute_objective, "Simple bqm")
# visualize_graph(n, Delta, points, selected_medoids_bqm_fixed, alpha, beta, compute_objective, "Fixed roof duality")

draw_chart_obj_fun(selected_medoids_bqm, compute_objective, figure_title="Simple bqm")
draw_chart_obj_fun(selected_medoids_bqm_fixed, compute_objective, figure_title="Fixed bqm")

# Plot the selected medoids on the map
plot_medoids_on_map(df_5g, df_taxi, selected_medoids_bqm[0], "Simple_bqm_medoids")
plot_medoids_on_map(df_5g, df_taxi, selected_medoids_bqm_fixed[0], "Fixed_bqm_medoids")

plt.show()
