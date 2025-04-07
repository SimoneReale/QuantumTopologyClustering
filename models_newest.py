import dimod
import numpy as np
from create_dataset import create_pechino_dataset
import matplotlib.pyplot as plt
from dimod import ConstrainedQuadraticModel, quicksum




def check_term_balance(objective_values):
    """
    Analyzes and prints balance between penalty, spread, and constraint terms.
    
    Input: 
      - objective_values: dict from compute_objective()
      
    Output:
      - Prints balance ratios and suggests tweaks
    """

    # Extract terms
    penalty = abs(objective_values["penalty_term"])
    spread = abs(objective_values["spread_term"])
    constraint = abs(objective_values["constraint_violation"])
    
    # Normalize terms (avoid division by zero)
    total = penalty + spread + constraint + 1e-6
    penalty_ratio = penalty / total
    spread_ratio = spread / total
    constraint_ratio = constraint / total

    print(f"Relative Contributions:")
    print(f"  - Penalty Term:    {penalty_ratio:.2%}")
    print(f"  - Spread Term:     {spread_ratio:.2%}")
    print(f"  - Constraint Term: {constraint_ratio:.2%}")

    # Identify imbalances
    dominant = max(penalty_ratio, spread_ratio, constraint_ratio)
    if penalty_ratio == dominant:
        print("⚠️ Penalty domina! Stupido riduci penalty_weight.")
    elif spread_ratio == dominant:
        print("⚠️ Spread domina! Stupido riduci spread_weight.")
    elif constraint_ratio == dominant:
        print("⚠️ Constraint domina! Stupido riduci lagrange_multiplier.")
    
    # Check ignored terms
    if penalty_ratio < 0.05:
        print("⚠️ Penalty è debole! Aumenta penalty_weight.")
    if spread_ratio < 0.05:
        print("⚠️ Spread è debole! Aumenta spread_weight.")
    if constraint_ratio < 0.05:
        print("⚠️ Constraint è debole! Aumenta lagrange_multiplier.")

    return penalty_ratio, spread_ratio, constraint_ratio


def choose_weights(n, k, Delta, min_distance_normalized, c_p=1.0, c_s=1.0):
    """Automatically chooses penalty and spread weights based on n, k, and distance matrix."""
    avg_distance = np.mean(Delta)  # Average pairwise distance
    # penalty_weight = 100 * ((c_p * avg_distance) / k)
    # #penalty_weight = (c_p * avg_distance) / (k * Delta.max())

    # spread_weight = (c_s * avg_distance) / np.sqrt(n)


    num_pairs_penalty = np.sum(Delta < min_distance_normalized)  # Quante coppie sono sotto la soglia
    num_pairs_spread = np.sum(Delta >= min_distance_normalized)  # Quante coppie sono sopra

    ratio = num_pairs_spread / (num_pairs_penalty + 1e-6)  # Evita divisioni per zero

    penalty_weight = 50 * (c_p * avg_distance / k) * ratio
    spread_weight = (c_s * avg_distance) / np.sqrt(n)

    return penalty_weight, spread_weight


def calculate_lambda(importance_values, penalty_weight, spread_weight, k, importance_strength=1.0):
    """
    Calculates the importance weight (lambda_) based on heuristic balancing.

    Args:
        importance_values: Array of importance values for each point.
        penalty_weight: Calculated spatial penalty weight (alpha).
        spread_weight: Calculated spatial spread weight (beta).
        k: Number of points to select.
        importance_strength: User-tunable factor controlling relative importance 
                             (default: 1.0 attempts balance).

    Returns:
        float: The calculated lambda_ value.
    """
    mean_abs_importance = np.mean(np.abs(importance_values))
    # Handle case where all importances are zero or near zero
    if mean_abs_importance < 1e-9:
        return 0.0 # Importance has no effect if values are zero

    avg_spatial_weight = (penalty_weight + spread_weight) / 2.0
    
    # Heuristic: Balance avg linear importance with avg quadratic spatial interaction per node
    # lambda * mean_abs_imp = C * (k-1) * avg_spatial_weight 
    # (k-1 is the approx number of spatial interactions per selected node)
    lambda_val = importance_strength * (k - 1) * avg_spatial_weight / mean_abs_importance
        
    return lambda_val

def create_bqm_even_spread(n, k, Delta, min_distance, lagrange_multiplier, importance_values, c_p=1.0, c_s=1.0, lambda_=1.0):
    """
    Creates a BQM that ensures an even spatial spread of selected medoids 
    and provides an objective function evaluator.

    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix (n x n)
    - min_distance: Minimum allowed distance between medoids
    - lagrange_multiplier: Strength of the constraint enforcing exactly k medoids
    - importance_values: Array of importance values for each point
    - c_p: Coefficient for penalty weight selection
    - c_s: Coefficient for spread weight selection
    - lambda_: Weight for importance biasing

    Returns:
    - bqm: The Binary Quadratic Model
    - compute_objective: A function to evaluate penalties, rewards, and constraints given a solution.
    """
    # Normalize min_distance between 0 and 1
    min_distance_normalized = min_distance / Delta.max()

    Delta = Delta / Delta.max()

    # Auto-select weights
    penalty_weight, spread_weight = choose_weights(n, k, Delta, min_distance_normalized, c_p, c_s)

    bqm = dimod.BinaryQuadraticModel('BINARY')
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    # Objective function components
    quadratic = {}

    def compute_objective(selected_medoids, Delta, importance_values):
        """Computes objective function components for a given set of selected medoids."""
        # Normalize Delta between 0 and 1
        Delta = Delta / Delta.max()
        penalty_term = 0
        spread_term = 0
        importance_term = 0
        constraint_violation = 0

        num_selected = len(selected_medoids)

        # Compute penalties and rewards
        for i in selected_medoids:
            for j in selected_medoids:
                if i < j:
                    distance = Delta[i, j]
                    if distance < min_distance_normalized:
                        penalty_term += penalty_weight * (min_distance_normalized - distance)
                    else:
                        spread_term += -spread_weight * distance
        
        # Compute importance bias
        importance_term = -lambda_ * sum(importance_values[i] for i in selected_medoids)

        # Compute constraint violation
        constraint_violation = lagrange_multiplier * (num_selected - k) ** 2

        total_objective = penalty_term + spread_term + importance_term + constraint_violation
        dict_objective = {
            "total_objective": total_objective,
            "penalty_term": penalty_term,
            "spread_term": spread_term,
            "importance_term": importance_term,
            "constraint_violation": constraint_violation
        }

        calculate_all_terms(n, k, Delta, min_distance_normalized, importance_values, c_p, c_s, lambda_, lagrange_multiplier)
        
        return dict_objective

    # Build the BQM
    for i in range(n):
        for j in range(i + 1, n):
            distance = Delta[i, j]
            if distance < min_distance_normalized:
                quadratic[(f'z_{i}', f'z_{j}')] = penalty_weight * (min_distance_normalized - distance)
            else:
                quadratic[(f'z_{i}', f'z_{j}')] = -spread_weight * distance

    # Add quadratic terms to BQM
    bqm.add_quadratic_from(quadratic)

    # Add importance bias
    for i in range(n):
        bqm.add_linear(f'z_{i}', -lambda_ * importance_values[i])

    # Constraint: Select exactly k medoids
    bqm.add_linear_equality_constraint(
        [(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=lagrange_multiplier
    )

    print("\nThis are the terms of the BQM:")
    calculate_all_terms(n, k, Delta, min_distance_normalized, importance_values, c_p, c_s, lambda_, lagrange_multiplier)

    return bqm, compute_objective

def create_bqm_p_median(n, k, Delta, min_distance, lagrange_multiplier, importance_values, demand_assign=1.0, selection_constr=1.0, lambda_=1.0):
    """
    Creates a BQM for the p-Median problem.

    Parameters:
    - n: Number of potential service (facility) locations
    - k: Number of medoids
    - Delta: Distance matrix (n x n) between service locations and demand points
    - min_distance: Minimum allowed distance between medoids
    - lagrange_multiplier: Strength of the constraint enforcing exactly k medoids
    - importance_values: Array of importance values for each point
    - c_p: Penalty coefficient for demand assignment constraint
    - c_s: Penalty coefficient for selection constraint
    - lambda_: Weight for importance biasing

    Returns:
    - bqm: The Binary Quadratic Model
    - compute_objective: A function to evaluate penalties, rewards, and constraints given a solution.
    """
    bqm = dimod.BinaryQuadraticModel('BINARY')

    # Normalize min_distance between 0 and 1
    min_distance_normalized = min_distance / Delta.max()
    Delta = Delta / Delta.max()

    # Decision variables
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}
    x = {(i, j): dimod.Binary(f'x_{i}_{j}') for i in range(n) for j in range(n)}

    # Objective function: Minimize total assignment cost
    for i in range(n):
        for j in range(n):
            bqm.add_linear(f'x_{i}_{j}', Delta[i, j])

    # Constraint: Each demand point must be assigned to exactly one service location
    for j in range(n):
        bqm.add_linear_equality_constraint(
            [(f'x_{i}_{j}', 1) for i in range(n)], constant=-1, lagrange_multiplier=demand_assign
        )

    # Constraint: A demand point can only be assigned to an active service location
    for i in range(n):
        for j in range(n):
            bqm.add_quadratic(f'x_{i}_{j}', f'z_{i}', -selection_constr)

    # Constraint: Select exactly k medoids
    bqm.add_linear_equality_constraint(
        [(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=lagrange_multiplier
    )

    # Add importance bias
    for i in range(n):
        bqm.add_linear(f'z_{i}', -lambda_ * importance_values[i])

    def compute_objective(selected_medoids, Delta, importance_values):
        """Computes objective function components for a given set of selected medoids."""
        # Normalize Delta between 0 and 1
        Delta = Delta / Delta.max()
        penalty_term = 0
        spread_term = 0
        importance_term = 0
        constraint_violation = 0

        num_selected = len(selected_medoids)

        # Compute penalties and rewards
        for i in selected_medoids:
            for j in selected_medoids:
                if i < j:
                    distance = Delta[i, j]
                    if distance < min_distance_normalized:
                        penalty_term += demand_assign * (min_distance_normalized - distance)
                    else:
                        spread_term += -selection_constr * distance
        
        # Compute importance bias
        importance_term = -lambda_ * sum(importance_values[i] for i in selected_medoids)

        # Compute constraint violation
        constraint_violation = lagrange_multiplier * (num_selected - k) ** 2

        total_objective = penalty_term + spread_term + importance_term + constraint_violation
        dict_objective = {
            "total_objective": total_objective,
            "penalty_term": penalty_term,
            "spread_term": spread_term,
            "importance_term": importance_term,
            "constraint_violation": constraint_violation
        }

        return dict_objective

    return bqm, compute_objective

def calculate_all_terms(n, k, Delta, min_distance, importance_values, c_p=1.0, c_s=1.0, lambda_=1.0, lagrange_multiplier=1.0):
    """
    Calculates the sum of all penalties, spread terms, importance terms, and the constraint term if all medoids are chosen.

    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix (n x n)
    - min_distance: Minimum allowed distance between medoids
    - importance_values: Array of importance values for each point
    - c_p: Coefficient for penalty weight selection
    - c_s: Coefficient for spread weight selection
    - lambda_: Weight for importance biasing

    Returns:
    - total_penalty: Sum of all penalties
    - total_spread: Sum of all spread terms
    - total_importance: Sum of all importance terms
    - constraint_term: Constraint term for selecting all medoids
    """
    print("\nCalculating all terms...")

    # Normalize min_distance between 0 and 1
    min_distance_normalized = min_distance / Delta.max()
    Delta = Delta / Delta.max()

    # Auto-select weights
    penalty_weight, spread_weight = choose_weights(n, k, Delta, min_distance_normalized, c_p, c_s)

    total_penalty = 0
    total_spread = 0
    total_importance = 0
    max_penalty = float('-inf')
    min_penalty = float('inf')
    max_spread = float('-inf')
    min_spread = float('inf')
    max_importance = float('-inf')
    min_importance = float('inf')

    # Calculate penalties, spread terms, and importance terms
    for i in range(n):
        for j in range(i + 1, n):
            distance = Delta[i, j]
            if distance < min_distance_normalized:
                penalty = penalty_weight * (min_distance_normalized - distance)
                total_penalty += penalty
                max_penalty = max(max_penalty, penalty)
                min_penalty = min(min_penalty, penalty)
            else:
                spread = -spread_weight * distance
                total_spread += spread
                max_spread = max(max_spread, spread)
                min_spread = min(min_spread, spread)

    for i in range(n):
        importance = -lambda_ * importance_values[i]
        total_importance += importance
        max_importance = max(max_importance, importance)
        min_importance = min(min_importance, importance)

    # Constraint term
    constraint_term = lagrange_multiplier * (n - k) ** 2

    print(f"Total Penalty: {total_penalty}")
    print(f"Max Penalty: {max_penalty}")
    print(f"Min Penalty: {min_penalty}")
    print(f"Total Spread: {total_spread}")
    print(f"Max Spread: {max_spread}")
    print(f"Min Spread: {min_spread}")
    print(f"Total Importance: {total_importance}")
    print(f"Max Importance: {max_importance}")
    print(f"Min Importance: {min_importance}")
    print(f"Constraint Term: {constraint_term}")

    return total_penalty, total_spread, total_importance, constraint_term

def create_cqm_even_spread(n, k, Delta, min_distance, lagrange_multiplier, importance_values, c_p=1.0, c_s=1.0, lambda_=1.0):
    """
    Creates a CQM that ensures an even spatial spread of selected medoids 
    and provides an objective function evaluator.

    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix (n x n)
    - min_distance: Minimum allowed distance between medoids
    - lagrange_multiplier: Strength of the constraint enforcing exactly k medoids
    - importance_values: Array of importance values for each point
    - c_p: Coefficient for penalty weight selection
    - c_s: Coefficient for spread weight selection
    - lambda_: Weight for importance biasing

    Returns:
    - cqm: The Constrained Quadratic Model
    - compute_objective: A function to evaluate penalties, rewards, and constraints given a solution.
    """
    # Normalize min_distance between 0 and 1
    min_distance_normalized = min_distance / Delta.max()
    Delta = Delta / Delta.max()

    # Auto-select weights
    penalty_weight, spread_weight = choose_weights(n, k, Delta, min_distance_normalized, c_p, c_s)

    # Create the CQM
    cqm = ConstrainedQuadraticModel()

    # Decision variables
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    # Objective function components
    penalty_term = quicksum(
        penalty_weight * (min_distance_normalized - Delta[i, j]) * z[i] * z[j]
        for i in range(n) for j in range(i + 1, n) if Delta[i, j] < min_distance_normalized
    )
    spread_term = quicksum(
        -spread_weight * Delta[i, j] * z[i] * z[j]
        for i in range(n) for j in range(i + 1, n) if Delta[i, j] >= min_distance_normalized
    )
    importance_term = quicksum(-lambda_ * importance_values[i] * z[i] for i in range(n))

    # Set the objective
    cqm.set_objective(penalty_term + spread_term + importance_term)

    # Constraint: Select exactly k medoids
    cqm.add_constraint(quicksum(z[i] for i in range(n)) == k, label="select_k_medoids")

    # Function to compute objective components
    def compute_objective(selected_medoids, Delta, importance_values):
        """Computes objective function components for a given set of selected medoids."""
        # Normalize Delta between 0 and 1
        Delta = Delta / Delta.max()
        penalty_term = 0
        spread_term = 0
        importance_term = 0
        constraint_violation = 0

        num_selected = len(selected_medoids)

        # Compute penalties and rewards
        for i in selected_medoids:
            for j in selected_medoids:
                if i < j:
                    distance = Delta[i, j]
                    if distance < min_distance_normalized:
                        penalty_term += penalty_weight * (min_distance_normalized - distance)
                    else:
                        spread_term += -spread_weight * distance

        # Compute importance bias
        importance_term = -lambda_ * sum(importance_values[i] for i in selected_medoids)

        # Compute constraint violation
        constraint_violation = lagrange_multiplier * (num_selected - k) ** 2

        total_objective = penalty_term + spread_term + importance_term + constraint_violation
        dict_objective = {
            "total_objective": total_objective,
            "penalty_term": penalty_term,
            "spread_term": spread_term,
            "importance_term": importance_term,
            "constraint_violation": constraint_violation
        }

        return dict_objective

    return cqm, compute_objective

# if __name__ == "__main__":
#     filter_radius = 5000
#     Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(filter_radius)
#     min_distance = 3000
#     k = 20
#     min_distance = min_distance / Delta.max()
#     Delta = Delta / Delta.max()
#     calculate_all_terms(n, k, Delta, min_distance, importance_values, c_p=1, c_s=1, lambda_=0.1, lagrange_multiplier=0.2)
#     penalty_weight, spread_weight = choose_weights(n, k, Delta, min_distance, c_p=1, c_s=1)
#     print("\n\n")
#     calculate_all_terms(n, k, Delta, min_distance, importance_values, c_p=1, c_s=1, lambda_=calculate_lambda(importance_values, penalty_weight, spread_weight, k, 1), lagrange_multiplier=0.2)

import time
if __name__ == "__main__":
    times = []

    filter_radius = 4000
    Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(filter_radius)




    for _ in range(5):

        start_time = time.time()

        bqm, _ = create_bqm_even_spread(
        n,
        20,
        Delta,
        3000,
        c_p=1,
        c_s=1,
        lambda_=1,
        lagrange_multiplier=2,
        importance_values=importance_values,
    )
        print("Bqm num of bytes: ", bqm.nbytes())

        end_time = time.time()
        dataset_creation_time = end_time - start_time
        print("Dataset creation time: ", dataset_creation_time)
        times.append(dataset_creation_time)
    # create_cqm_even_spread(n, 20, Delta, 300, 1)
    #create_bqm_only_penalty(n, 20, Delta, 3000, importance_values, alpha=100, lambda_=0.01)
    print(f"Average time to create BQM: {np.mean(times)} seconds Standard deviation: {np.std(times)} seconds")