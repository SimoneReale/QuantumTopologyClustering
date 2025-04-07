from dwave.preprocessing.presolve import Presolver
from dwave.preprocessing.lower_bounds import roof_duality
import numpy as np
import dimod
from create_dataset import create_pechino_dataset



def create_cqm(n, k, alpha, beta, Delta, importance_values, lambda_=0.1):
    """
    Creates a Constrained Quadratic Model (CQM) for the k-medoids problem with bias towards important points.

    Parameters:
    - n: Number of data points
    - k: Number of medoids
    - alpha: Tradeoff parameter for dispersion
    - beta: Tradeoff parameter for centrality
    - Delta: Similarity matrix
    - importance_values: Array of importance values for each point

    Returns:
    - cqm: The CQM model
    - compute_objective: Function to compute the objective value for a given set of medoids
    """
    cqm = dimod.ConstrainedQuadraticModel()
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    dispersion = -alpha * sum(Delta[i, j] * z[i] * z[j] for i in range(n) for j in range(n) if i != j)
    centrality = beta * sum(Delta[i, j] * z[i] for i in range(n) for j in range(n))
    #importance_bias = -lambda_ * sum(importance_values[i] * z[i] for i in range(n))

    #cqm.set_objective(dispersion + centrality + importance_bias)
    cqm.set_objective(dispersion + centrality)

    # Constraint: exactly k medoids 
    cqm.add_constraint(sum(z[i] for i in range(n)) == k, label='select_k_medoids')

    def compute_objective(selected_medoids):
        """
        Computes the objective function value for a given set of medoids.

        Parameters:
        - selected_medoids: List of indices representing the selected medoids

        Returns:
        - Objective function value
        """
        dispersion_val = -alpha * sum(Delta[i, j] for i in selected_medoids for j in selected_medoids if i != j)
        centrality_val = beta * sum(Delta[i, j] for i in selected_medoids for j in range(n))
        importance_val = -lambda_ * sum(importance_values[i] for i in selected_medoids)
        return dispersion_val, centrality_val, importance_val

    return cqm, compute_objective

def create_native_bqm(n, k, alpha, beta, Delta, lagrange_multiplier):
    """
    Creates a native Binary Quadratic Model (BQM) for the k-medoids problem with bias towards important points.

    Parameters:
    - n: Number of data points
    - k: Number of medoids
    - alpha: Tradeoff parameter for dispersion
    - beta: Tradeoff parameter for centrality
    - Delta: Similarity matrix
    - importance_values: Array of importance values for each point

    Returns:
    - bqm: The BQM model
    """
    #normalizzo delta
    Delta = Delta / Delta.max()

    bqm = dimod.BinaryQuadraticModel('BINARY')
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    # Objective function
    dispersion = -alpha * sum(Delta[i, j] * z[i] * z[j] for i in range(n) for j in range(n) if i != j)
    centrality = beta * sum(Delta[i, j] * z[i] for i in range(n) for j in range(n))

    objective = dispersion + centrality

    for i in range(n):
        bqm.add_variable(f'z_{i}', objective.linear[f'z_{i}'])
    for (i, j), bias in objective.quadratic.items():
        bqm.add_interaction(f'z_{i}', f'z_{j}', bias)

    # Constraint: exactly k medoids
    bqm.add_linear_equality_constraint([(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=lagrange_multiplier)

    return bqm


import time
if __name__ == "__main__":
    times = []

    filter_radius = 8000
    Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(filter_radius)
    k = 20
    alpha = 1 / k
    beta = 1 / n
    lagrange_multiplier = 2


    for _ in range(5):
        start_time = time.time()
        bqm = create_native_bqm(n, k, alpha, beta, Delta, lagrange_multiplier)
        print("Bqm num of bytes: ", bqm.nbytes())
        end_time = time.time()
        dataset_creation_time = end_time - start_time
        print("Dataset creation time: ", dataset_creation_time)
        times.append(dataset_creation_time)
    # create_cqm_even_spread(n, 20, Delta, 300, 1)
    #create_bqm_only_penalty(n, 20, Delta, 3000, importance_values, alpha=100, lambda_=0.01)
    print(f"Average time to create BQM: {np.mean(times)} seconds Standard deviation: {np.std(times)} seconds")



