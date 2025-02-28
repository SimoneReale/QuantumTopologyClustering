import dimod
from dwave.preprocessing.presolve import Presolver
from dwave.preprocessing.lower_bounds import roof_duality

def create_cqm(n, k, alpha, beta, Delta, importance_values):
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
    importance_bias = sum(importance_values[i] * z[i] for i in range(n))

    cqm.set_objective(dispersion + centrality + importance_bias)

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
        importance_val = sum(importance_values[i] for i in selected_medoids)
        return dispersion_val, centrality_val, importance_val

    return cqm, compute_objective



def create_bqm(cqm):
    bqm, invert = dimod.cqm_to_bqm(cqm)
    print(f"Interactions: {bqm.num_interactions} Var: {bqm.num_variables}")
    return bqm, invert
