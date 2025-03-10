from ortools.sat.python import cp_model

def create_ortools_model(n, k, alpha, beta, Delta, importance_values):
    """
    Creates a Constrained Quadratic Model (CQM) with OR-Tools CP-SAT solver for the k-medoids problem.

    Parameters:
    - n: Number of data points
    - k: Number of medoids
    - alpha: Tradeoff parameter for dispersion
    - beta: Tradeoff parameter for centrality
    - Delta: Similarity matrix
    - importance_values: Array of importance values for each point

    Returns:
    - model: The OR-Tools CP-SAT model
    """
    model = cp_model.CpModel()
    z = [model.NewBoolVar(f'z_{i}') for i in range(n)]

    # Objective function
    dispersion_terms = []
    for i in range(n):
        for j in range(n):
            if i != j:
                dispersion_var = model.NewIntVar(0, 1, f'dispersion_{i}_{j}')
                model.AddMultiplicationEquality(dispersion_var, z[i], z[j])
                dispersion_terms.append(-alpha * Delta[i, j] * dispersion_var)
    dispersion = sum(dispersion_terms)

    centrality_terms = []
    for i in range(n):
        for j in range(n):
            centrality_terms.append(beta * Delta[i, j] * z[i])
    centrality = sum(centrality_terms)

    importance_bias_terms = []
    for i in range(n):
        importance_bias_terms.append(-importance_values[i] * z[i])
    importance_bias = sum(importance_bias_terms)

    model.Minimize(dispersion + centrality + importance_bias)

    # Constraint: exactly k medoids
    model.Add(sum(z[i] for i in range(n)) == k)

    return model