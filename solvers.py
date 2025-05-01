from alive_progress import alive_bar
from dimod import SimulatedAnnealingSampler, ExactCQMSolver
from dwave.preprocessing.composites import FixVariablesComposite
from dwave.preprocessing.presolve import Presolver
from ortools.sat.python import cp_model
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridCQMSampler
from config import tokenuccio
import pulp
from sklearn_extra.cluster import KMedoids
import dwave.inspector

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
        sampler = EmbeddingComposite(
                    DWaveSampler(
                        token=tokenuccio,
                        endpoint="https://na-west-1.cloud.dwavesys.com/sapi/v2/",
                        solver="Advantage_system7.1",
                        chain_strength=3000.0,
                    ))

        solution_bqm = sampler.sample(bqm, num_reads=1000)
        #dwave.inspector.show(solution_bqm)
        selected_medoids_bqm = [[i for i in range(num_of_nodes) if sample[f'z_{i}'] == 1] for sample in solution_bqm.samples()][:num_of_sol]
        bar()
    return selected_medoids_bqm

def hybrid_cqm_solve(cqm, num_of_nodes, num_of_sol):
    with alive_bar(1) as bar:
        print("Hybrid CQM solver")
        sampler = LeapHybridCQMSampler(token=tokenuccio)
        result = sampler.sample_cqm(cqm)
        feasible_samples = result.filter(lambda d: d.is_feasible)
        selected_medoids = [[i for i in range(num_of_nodes) if sample[f'z_{i}'] == 1] for sample in feasible_samples.samples()[:num_of_sol]]
        bar()
    return selected_medoids

def p_median_kmedoids(df_taxi, df_5g, k, importance_values, gamma=1.0):
    """
    Solves the k-medoids problem using the p-Median model with PuLP.

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
            Delta[i, j] = np.sqrt((df_5g.iloc[i]['lat'] - df_taxi.iloc[j]['lat'])**2 + (df_5g.iloc[i]['lon'] - df_taxi.iloc[j]['lon'])**2)

    # Create the modified distance matrix incorporating importance values
    modified_Delta = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            modified_Delta[i, j] = Delta[i, j] / (importance_values[i] ** gamma)

    # Create the PuLP problem
    prob = pulp.LpProblem("IMportance P-Median", pulp.LpMinimize)

    # Decision variables
    z = pulp.LpVariable.dicts('z', range(n), cat='Binary')
    x = pulp.LpVariable.dicts('x', [(i, j) for i in range(n) for j in range(m)], cat='Binary')

    # Objective function: Minimize total assignment cost
    prob += pulp.lpSum(modified_Delta[i, j] * x[i, j] for i in range(n) for j in range(m))

    # Constraint: Each demand point must be assigned to exactly one service location
    for j in range(m):
        prob += pulp.lpSum(x[i, j] for i in range(n)) == 1

    # Constraint: A demand point can only be assigned to an active service location
    for i in range(n):
        for j in range(m):
            prob += x[i, j] <= z[i]

    # Constraint: Select exactly k medoids
    prob += pulp.lpSum(z[i] for i in range(n)) == k

    # Solve the problem
    with alive_bar(1) as bar:
        print("Solving p-Median model with PuLP")
        prob.solve()
        selected_medoids = [i for i in range(n) if pulp.value(z[i]) == 1]
        bar()

    print("selected medoids: ", selected_medoids)

    return selected_medoids

def sklearn_kmedoids(df_5g, k):
    """
    Solves the k-medoids problem using the k-medoids algorithm from scikit-learn.

    Parameters:
    - df_5g: DataFrame containing the 5G antenna data (service points)
    - k: Number of medoids

    Returns:
    - selected_medoids: List of selected medoids
    """

    # Fit the k-medoids model
    kmedoids = KMedoids(n_clusters=k, random_state=0).fit(df_5g[['lat', 'lon']].values)

    # Get the indices of the selected medoids
    selected_medoids = kmedoids.medoid_indices_

    selected_medoids = selected_medoids.tolist()

    print("selected medoids: ", selected_medoids)

    return selected_medoids



