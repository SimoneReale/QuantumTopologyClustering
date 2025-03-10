from alive_progress import alive_bar
from dimod import SimulatedAnnealingSampler, ExactCQMSolver
from dwave.preprocessing.composites import FixVariablesComposite
from dwave.preprocessing.presolve import Presolver
from ortools.sat.python import cp_model
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite

def simple_simulated_annealing(bqm, num_of_nodes, num_of_sol):
    with alive_bar(1) as bar:   
        print("No preprocessing")
        sampler = SimulatedAnnealingSampler()
        solution_bqm = sampler.sample(bqm)
        selected_medoids_bqm = [[i for i in range(num_of_nodes) if sample[f'z_{i}'] == 1] for sample in solution_bqm.samples()][:num_of_sol]
        bar()
    return selected_medoids_bqm

def roof_duality_simulated_annealing(bqm, num_of_nodes, num_of_sol):
    with alive_bar(1) as bar:
        print("Fixed roof duality")
        sampler_fixed = FixVariablesComposite(SimulatedAnnealingSampler(), algorithm='roof_duality')
        solution_bqm_fixed = sampler_fixed.sample(bqm)
        selected_medoids_bqm_fixed = [[i for i in range(num_of_nodes) if sample[f'z_{i}'] == 1] for sample in solution_bqm_fixed.samples()][:num_of_sol]
        bar()
    return selected_medoids_bqm_fixed

def exact_cqm_solve(cqm, num_of_nodes, num_of_sol):
    with alive_bar(1) as bar:
        print("Exact CQM solver")
        sampler = ExactCQMSolver() 
        result = sampler.sample_cqm(cqm)
        feasible_samples = result.filter(lambda d: d.is_feasible)
        selected_medoids = [[i for i in range(num_of_nodes) if sample[f'z_{i}'] == 1] for sample in feasible_samples.samples()[:num_of_sol]]
        bar()
    return selected_medoids

def ortools_cqm_solve(model, num_of_nodes):

    solver = cp_model.CpSolver()

    with alive_bar(1) as bar:
        print("Solving CQM with OR-Tools CP-SAT solver")
        status = solver.Solve(model)
        bar()

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        selected_medoids = [i for i in range(num_of_nodes) if solver.Value(z[i]) == 1]
        print("Ortools found a feasible solution: ", selected_medoids)
    else:
        selected_medoids = []

    return selected_medoids

def greedy_kmedoids(n, k, Delta, importance_values, bias=True):
    """
    Solves the k-medoids problem using a greedy algorithm with bias towards important points.

    Parameters:
    - n: Number of data points
    - k: Number of medoids
    - Delta: Similarity matrix
    - importance_values: Array of importance values for each point

    Returns:
    - selected_medoids: List of selected medoids
    """
    selected_medoids = []
    remaining_points = set(range(n))

    first_medoid = np.random.choice(list(remaining_points))
    selected_medoids.append(first_medoid)
    remaining_points.remove(first_medoid)

    with alive_bar(k - 1) as bar:
        print("Greedy algorithm")
        for _ in range(k - 1):
            if(bias):
                next_medoid = max(remaining_points, key=lambda x: (min(Delta[x, m] + importance_values[x] for m in selected_medoids)))
            else:
                next_medoid = max(remaining_points, key=lambda x: (min(Delta[x, m]  for m in selected_medoids)))
            selected_medoids.append(next_medoid)
            remaining_points.remove(next_medoid)
            bar()

    print("selected medoids: ", selected_medoids)

    return selected_medoids

def quantum_annealing(bqm, num_of_nodes, num_of_sol):
    with alive_bar(1) as bar:
        print("Quantum annealing")
        sampler = EmbeddingComposite(DWaveSampler())
        solution_bqm = sampler.sample(bqm, num_reads=1000)
        selected_medoids_bqm = [[i for i in range(num_of_nodes) if sample[f'z_{i}'] == 1] for sample in solution_bqm.samples()][:num_of_sol]
        bar()
    return selected_medoids_bqm



