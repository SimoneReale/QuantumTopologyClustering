import dimod
from dimod import ConstrainedQuadraticModel, Binary, quicksum
import numpy as np
from create_dataset import create_pechino_dataset
import matplotlib.pyplot as plt
from dimod import ConstrainedQuadraticModel, Binary, quicksum


def choose_weights_sophisticated(n, k, Delta, min_distance, importance_values, alpha=1.0, lambda_=1.0):
    """
    More sophisticated method to determine penalty and constraint weights based on problem characteristics.
    
    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix (n x n)
    - min_distance: Minimum allowed distance between medoids
    - importance_values: Array of importance values for each point
    - alpha: Base weight for the constraint enforcement
    - lambda_: Base weight for importance contribution

    Returns:
    - c_p: Penalty weight for distance enforcement
    - alpha: Adjusted weight for constraint enforcement
    - lambda_: Adjusted weight for importance biasing
    """
    avg_distance = np.mean(Delta)
    max_distance = np.max(Delta)
    min_distance_ratio = min_distance / max_distance  # Normalize min_distance

    print(f"Avg distance: {avg_distance}, Min distance: {min_distance}, Max distance: {max_distance}")

    # Set penalty weight dynamically
    c_p = (avg_distance ** 2) / (min_distance ** 2)  

    # Adjust alpha dynamically based on n, k, and difficulty of satisfying constraints
    alpha = alpha * (n / k) ** 1.5  

    # Adjust lambda_ based on importance value distribution (normalized)
    importance_range = np.ptp(importance_values)  # Peak-to-peak (max-min)
    lambda_ = lambda_ * (importance_range / np.mean(importance_values) if np.mean(importance_values) > 0 else 1)

    return c_p, alpha, lambda_


def choose_weights_sophisticated_2(n, k, Delta, min_distance, importance_values, alpha=1.0, beta = 1.0, lambda_=1.0):
    """
    Improved method to determine penalty and constraint weights based on problem characteristics.
    
    Adjusts penalty to avoid being too dominant and increases the influence of importance values.
    """
    avg_distance = np.mean(Delta)
    max_distance = np.max(Delta)

    min_distance_ratio = min_distance / max_distance  # Normalize min_distance

    print(f"Avg distance: {avg_distance}, Min distance: {min_distance}, Max distance: {max_distance}")

    # REDUCE PENALTY STRENGTH: Weaken c_p so it allows importance influence
    c_p = beta * (avg_distance / min_distance)  

    # ADJUST CONSTRAINT WEIGHT (alpha) dynamically
    alpha = alpha * (n / k) ** 1.2  # Reduce exponent to make it slightly weaker

    # BOOST IMPORTANCE WEIGHT: Increase importance contribution
    importance_std = np.std(importance_values)
    lambda_ = lambda_ * (importance_std / np.mean(importance_values) if np.mean(importance_values) > 0 else 1) * 2  

    return c_p, alpha, lambda_




def choose_weights_balanced_hist(n, k, Delta, min_distance, alpha=0.5, beta=0.5):
    """
    Selects penalty_weight and spread_weight using histogram-based balancing.

    - Uses number of violating and non-violating antenna pairs to adjust relative strength.
    - Ensures penalty and spread terms have comparable magnitudes.

    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix (n x n)
    - min_distance: Minimum allowed distance between medoids
    - alpha: Scaling factor for penalty term (0 to 1)
    - beta: Scaling factor for spread term (0 to 1)

    Returns:
    - penalty_weight (c_p)
    - spread_weight (c_s)
    """

    min_distance_normalized = min_distance / Delta.max()

    # Count number of pairs violating the minimum distance
    N_penalty = np.sum(Delta < min_distance_normalized) / 2  # Since Delta is symmetric

    # Count number of pairs that are correctly spaced apart
    N_spread = np.sum(Delta >= min_distance_normalized) / 2  

    print(f"Pairs violating min distance: {N_penalty}, Pairs correctly spaced: {N_spread}")

    # Normalize importance of both terms
    penalty_weight = alpha * (N_spread / (N_penalty + 1e-6)) * (min_distance / np.mean(Delta))
    spread_weight = beta * (N_penalty / (N_spread + 1e-6)) * (np.mean(Delta) / min_distance)

    return penalty_weight, spread_weight



def create_bqm_only_penalty(n, k, Delta, min_distance, importance_values, alpha=1.0, beta = 1.0, lambda_=1.0):
    """
    Creates a BQM that prioritizes satisfying the minimum distance constraint and selecting important medoids.
    
    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix (n x n)
    - min_distance: Minimum allowed distance between medoids
    - importance_values: Array of importance values for each point
    - alpha: Lagrange multiplier for k-medoids constraint
    - lambda_: Weight for importance biasing
    
    Returns:
    - bqm: The Binary Quadratic Model
    - compute_objective: A function to evaluate penalties, rewards, and constraints given a solution.
    """
    min_distance = min_distance / Delta.max()
    # Normalize distances
    Delta = Delta / np.max(Delta)
    
    # Auto-select weights
    c_p, alpha, lambda_ = choose_weights_sophisticated_3(n, k, Delta, min_distance, importance_values, alpha, lambda_)
    
    bqm = dimod.BinaryQuadraticModel('BINARY')
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    quadratic = {}

    def compute_objective(selected_medoids, Delta, importance_values):
        """Computes objective function components for a given set of selected medoids."""
        
        Delta = Delta / np.max(Delta)
        penalty_term = 0
        importance_term = 0
        constraint_violation = 0

        num_selected = len(selected_medoids)

        # Compute penalties
        for i in selected_medoids:
            for j in selected_medoids:
                if i < j:
                    distance = Delta[i, j]
                    penalty_term += c_p * (min_distance / distance) 
        
        # Compute importance bias
        importance_term = -lambda_ * sum(importance_values[i] for i in selected_medoids)
        
        # Compute constraint violation
        constraint_violation = alpha * (num_selected - k) ** 2

        total_objective = penalty_term + importance_term + constraint_violation
        return {
            "total_objective": total_objective,
            "penalty_term": penalty_term,
            "importance_term": importance_term,
            "constraint_violation": constraint_violation
        }

    # Build the BQM
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(f'z_{i}', f'z_{j}')] = c_p * (min_distance / Delta[i, j])  

    # Add quadratic terms to BQM
    bqm.add_quadratic_from(quadratic)

    # Add importance bias
    for i in range(n):
        bqm.add_linear(f'z_{i}', -lambda_ * importance_values[i])

    # Constraint: Select exactly k medoids
    bqm.add_linear_equality_constraint(
        [(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=alpha
    )

    return bqm, compute_objective



def choose_weights_sophisticated_3(n, k, Delta, min_distance, importance_values, alpha=0.5, beta=0.5, lambda_=0.5):
    """
    More sophisticated weight selection using statistics to ensure parameter sensitivity.
    
    - c_p is adjusted dynamically using statistical properties of Delta.
    - alpha is tuned based on n, k, and variance of distances.
    - lambda is normalized so importance values are on the same scale as other terms.

    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix (n x n)
    - min_distance: Minimum allowed distance between medoids
    - importance_values: Array of importance values for each point
    - alpha: Scaling factor for constraint enforcement (0 to 1)
    - beta: Scaling factor for spread enforcement (0 to 1)
    - lambda_: Scaling factor for importance contribution (0 to 1)

    Returns:
    - c_p: Penalty weight for enforcing minimum distance
    - alpha: Adjusted weight for constraint enforcement
    - lambda_: Adjusted weight for importance biasing
    """

    # Compute statistics
    avg_distance = np.mean(Delta)
    max_distance = np.max(Delta)
    min_distance_ratio = min_distance / max_distance
    std_distance = np.std(Delta)

    print(f"Avg: {avg_distance}, Min: {min_distance}, Max: {max_distance}, Std: {std_distance}")

    # PENALTY WEIGHT (c_p) - Spread enforcement  
    # Uses std deviation to scale sensitivity  
    c_p = beta * (std_distance / min_distance)

    # CONSTRAINT WEIGHT (alpha) - Ensure exactly k medoids  
    # Uses variance to adjust strength adaptively  
    alpha = alpha * ((n / k) ** 1.1) * (std_distance / avg_distance)

    # IMPORTANCE WEIGHT (lambda) - Favor high-importance antennas  
    # Normalizes importance values to fit within [0, 1]  
    importance_std = np.std(importance_values)
    importance_range = np.ptp(importance_values) if np.ptp(importance_values) > 0 else 1
    lambda_ = lambda_ * (importance_std / importance_range)  

    return c_p, alpha, lambda_




def create_cqm_even_spread(n, k, Delta, min_distance, importance, alpha=1.0, beta=1.0, lambda_=1.0):
    """
    Creates a CQM that ensures an even spatial spread of selected medoids, favors important antennas,
    and prioritizes maximizing the minimum distance between them.

    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix (n x n)
    - min_distance: Minimum allowed distance between medoids
    - importance: Array of importance values for each antenna (length n)
    - alpha, beta, lambda_: Weights for penalty, spread, and importance

    Returns:
    - cqm: The Constrained Binary Quadratic Model
    """

    nodes = set(range(n))
    indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Normalize distances
    min_distance_normalized = min_distance / Delta.max()
    Delta = Delta / Delta.max()

    # Normalize importance values (scale between 0 and 1)
    importance = np.array(importance)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)

    # Auto-select weights
    penalty_weight, spread_weight, importance_weight, _ = choose_weights(n, k, Delta, min_distance, importance, alpha, beta, lambda_)

    cqm = ConstrainedQuadraticModel()
    vars = {i: Binary(f"z_{i}") for i in range(n)}

    # Objective function
    obj = quicksum(
        (penalty_weight * (min_distance_normalized - Delta[i, j]) if Delta[i, j] < min_distance_normalized 
         else -spread_weight * Delta[i, j])
        * vars[i] * vars[j]
        for i, j in indices
    )

    # Add importance bias
    obj += quicksum(importance_weight * importance[i] * vars[i] for i in range(n))

    cqm.set_objective(obj)
    print("Objective function set.")

    # Constraint: Select exactly k medoids
    cqm.add_constraint(quicksum(vars[i] for i in range(n)) == k, label="Exact k medoids")
    print("Constraint OK: Exact k medoids")
    
    print("Model creation OK")
    return cqm

def check_term_balance(objective_values):
    """
    Analyzes and prints balance between penalty, spread, importance, and constraint terms.
    
    Input: 
      - objective_values: dict from compute_objective()
      
    Output:
      - Prints balance ratios and suggests tweaks
    """

    # Extract terms
    penalty = abs(objective_values["penalty_term"])
    spread = abs(objective_values["spread_term"])
    importance = abs(objective_values["importance_term"])
    constraint = abs(objective_values["constraint_violation"])
    
    # Normalize terms (avoid division by zero)
    total = penalty + spread + importance + constraint + 1e-6
    penalty_ratio = penalty / total
    spread_ratio = spread / total
    importance_ratio = importance / total
    constraint_ratio = constraint / total

    print(f"Relative Contributions:")
    print(f"  - Penalty Term:    {penalty_ratio:.2%}")
    print(f"  - Spread Term:     {spread_ratio:.2%}")
    print(f"  - Importance Term: {importance_ratio:.2%}")
    print(f"  - Constraint Term: {constraint_ratio:.2%}")

    # Identify imbalances
    dominant = max(penalty_ratio, spread_ratio, importance_ratio, constraint_ratio)
    if penalty_ratio == dominant:
        print("⚠️ Penalty domina! Stupido riduci penalty_weight.")
    elif spread_ratio == dominant:
        print("⚠️ Spread domina! Stupido riduci spread_weight.")
    elif importance_ratio == dominant:
        print("⚠️ Importance domina! Stupido riduci importance_weight.")
    elif constraint_ratio == dominant:
        print("⚠️ Constraint domina! Stupido riduci lagrange_multiplier.")
    
    # Check ignored terms
    if penalty_ratio < 0.05:
        print("⚠️ Penalty è debole! Aumenta penalty_weight.")
    if spread_ratio < 0.05:
        print("⚠️ Spread è debole! Aumenta spread_weight.")
    if importance_ratio < 0.05:
        print("⚠️ Importance è debole! Aumenta importance_weight.")
    if constraint_ratio < 0.05:
        print("⚠️ Constraint è debole! Aumenta lagrange_multiplier.")

    return penalty_ratio, spread_ratio, importance_ratio, constraint_ratio



def choose_weights(n, k, Delta, min_distance, importance, alpha, beta, lambda_):
    """Auto-calibrates weights to balance penalty, spread, importance, and constraints."""
    
    num_pairs_below = np.sum(Delta < min_distance / Delta.max())
    num_pairs_above = np.sum(Delta >= min_distance / Delta.max())

    # Set base weights relative to occurrences
    penalty_weight = alpha * (num_pairs_above / (num_pairs_below + 1e-6))  # Evita divisioni per zero
    spread_weight = beta * (num_pairs_below / (num_pairs_above + 1e-6))

    # Correzione automatica basata sui valori osservati
    penalty_weight *= 1e6  # Aumenta per renderlo influente
    spread_weight /= 10  # Riduci leggermente perché ancora troppo dominante
    importance_weight = lambda_ * 1e3  # Boost dell'importanza
    lagrange_multiplier = 1e3  # Rende più forte il vincolo
    
    return penalty_weight, spread_weight, importance_weight, lagrange_multiplier






def create_bqm_even_spread(n, k, Delta, min_distance, importance, alpha=1.0, beta=1.0, lambda_=1.0):
    """
    Creates a BQM that ensures an even spatial spread of selected medoids, 
    prioritizes important antennas, and enforces the k-medoids constraint.

    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix (n x n)
    - min_distance: Minimum allowed distance between medoids
    - importance: Array of importance values for each antenna
    - lagrange_multiplier: Strength of the constraint enforcing exactly k medoids
    - alpha, beta, lambda_: Weights for penalty, spread, and importance

    Returns:
    - bqm: The Binary Quadratic Model
    - compute_objective: A function to evaluate penalties, rewards, and constraints given a solution.
    """
    # Normalize distances
    min_distance_normalized = min_distance / Delta.max()
    Delta = Delta / Delta.max()

    # Normalize importance values (scale between 0 and 1)
    importance = np.array(importance)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)

    # Auto-select weights
    penalty_weight, spread_weight, importance_weight, lagrange_multiplier = choose_weights(n, k, Delta, min_distance, importance, alpha, beta, lambda_)

    bqm = dimod.BinaryQuadraticModel('BINARY')
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    # Objective function components
    quadratic = {}
    linear = {}

    def compute_objective(selected_medoids, Delta, importance):
        """Computes objective function components for a given set of selected medoids."""
        Delta = Delta / Delta.max()  # Normalize Delta between 0 and 1
        penalty_term = 0
        spread_term = 0
        importance_term = 0
        constraint_violation = 0

        num_selected = len(selected_medoids)

        # Compute penalties and rewards
        for i in selected_medoids:
            importance_term += importance_weight * importance[i]

            for j in selected_medoids:
                if i < j:
                    distance = Delta[i, j]
                    if distance < min_distance_normalized:
                        penalty_term += penalty_weight * (min_distance_normalized - distance)
                    else:
                        spread_term += -spread_weight * distance
        
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

        check_term_balance(dict_objective)
        return dict_objective

    # Build the BQM
    for i in range(n):
        # Add importance bias as a linear term
        linear[f'z_{i}'] = importance_weight * importance[i]

        for j in range(i + 1, n):
            distance = Delta[i, j]
            if distance < min_distance_normalized:
                quadratic[(f'z_{i}', f'z_{j}')] = penalty_weight * (min_distance_normalized - distance)
            else:
                quadratic[(f'z_{i}', f'z_{j}')] = -spread_weight * distance

    # Add terms to BQM
    bqm.add_linear_from(linear)
    bqm.add_quadratic_from(quadratic)

    # Constraint: Select exactly k medoids
    bqm.add_linear_equality_constraint(
        [(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=lagrange_multiplier
    )

    return bqm, compute_objective



import time 

if __name__ == "__main__":
    
    Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset(
            3000, taxi_data_file=f"splits/split_1.txt"
        )
    times = []
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
        end_time = time.time()
        dataset_creation_time = end_time - start_time
        times.append(dataset_creation_time)
    # create_cqm_even_spread(n, 20, Delta, 300, 1)
    #create_bqm_only_penalty(n, 20, Delta, 3000, importance_values, alpha=100, lambda_=0.01)
    print(f"Average time to create BQM: {np.mean(times)} seconds Standard deviation: {np.std(times)} seconds")