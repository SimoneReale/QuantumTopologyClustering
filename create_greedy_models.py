import dimod
import numpy as np

def create_cqm_greedy_kmedoids(n, k, Delta, importance_values, bias=True):
    """
    Creates a Constrained Quadratic Model (CQM) for the k-medoids problem that replicates the greedy algorithm.

    Parameters:
    - n: Number of data points
    - k: Number of medoids
    - Delta: Similarity matrix
    - importance_values: Array of importance values for each point
    - bias: Boolean indicating whether to use importance values in the selection

    Returns:
    - cqm: The CQM model
    """
    cqm = dimod.ConstrainedQuadraticModel()
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    # Objective function
    objective = dimod.QuadraticModel()
    for i in range(n):
        for j in range(n):
            if i != j:
                if bias:
                    objective.add_quadratic(f'z_{i}', f'z_{j}', -Delta[i, j] - importance_values[i])
                else:
                    objective.add_quadratic(f'z_{i}', f'z_{j}', -Delta[i, j])

    cqm.set_objective(objective)

    # Constraint: exactly k medoids
    cqm.add_constraint(sum(z[i] for i in range(n)) == k, label='select_k_medoids')

    return cqm

def create_bqm_greedy_kmedoids(n, k, Delta, importance_values, bias=True):
    """
    Creates a Binary Quadratic Model (BQM) for the k-medoids problem that replicates the greedy algorithm.

    Parameters:
    - n: Number of data points
    - k: Number of medoids
    - Delta: Similarity matrix
    - importance_values: Array of importance values for each point
    - bias: Boolean indicating whether to use importance values in the selection

    Returns:
    - bqm: The BQM model
    """
    bqm = dimod.BinaryQuadraticModel('BINARY')
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    # Objective function
    linear = {}
    quadratic = {}
    for i in range(n):
        linear[f'z_{i}'] = 0  # Initialize linear terms
        for j in range(n):
            if i != j:
                if bias:
                    quadratic[(f'z_{i}', f'z_{j}')] = -Delta[i, j] - importance_values[i]
                else:
                    quadratic[(f'z_{i}', f'z_{j}')] = -Delta[i, j]

    bqm.add_linear_from(linear)
    bqm.add_quadratic_from(quadratic)
    print(f"Delta min: {np.min(Delta)}, Delta max: {np.max(Delta)}")

    # Constraint: exactly k medoids
    bqm.add_linear_equality_constraint([(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=1)

    return bqm

def meters_to_degrees(meters):
    """
    Converts a distance in meters to degrees.

    Parameters:
    - meters: Distance in meters

    Returns:
    - degrees: Distance in degrees
    """
    min_distance_degrees = meters / 111320

    return 1 - np.exp(-0.5 * min_distance_degrees)

def create_bqm_max_min_distance(n, k, Delta, importance_values, min_distance, penalty_weight, bias=True):
    """
    Creates a Binary Quadratic Model (BQM) for the k-medoids problem that maximizes the minimum distance between medoids.

    Parameters:
    - n: Number of data points
    - k: Number of medoids
    - Delta: Similarity matrix (distance between points)
    - importance_values: Array of importance values for each point
    - min_distance: Minimum allowed distance between medoids in meters
    - penalty_weight: Weight for the penalty term
    - bias: Boolean indicating whether to use importance values in the selection

    Returns:
    - bqm: The BQM model
    """
    # Convert min_distance from meters to degrees
    min_distance_degrees = meters_to_degrees(min_distance)

    bqm = dimod.BinaryQuadraticModel('BINARY')
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    # Objective function
    linear = {}
    quadratic = {}
    
    for i in range(n):
        linear[f'z_{i}'] = 0
        for j in range(n):
            if i != j:
                # larger distances by using negative Delta
                if bias:
                    quadratic[(f'z_{i}', f'z_{j}')] = -1 * Delta[i, j] + importance_values[i]
                else:
                    quadratic[(f'z_{i}', f'z_{j}')] = -1 * Delta[i, j]

                # # Add strong penalty for being too close
                if Delta[i, j] < min_distance_degrees:
                    penalty = penalty_weight * (min_distance_degrees - Delta[i, j]) ** 2  # Quadratic penalty
                    print(f"Adding penalty {penalty} for distance {Delta[i, j]} < {min_distance_degrees}")
                    quadratic[(f'z_{i}', f'z_{j}')] += penalty

    bqm.add_linear_from(linear)
    bqm.add_quadratic_from(quadratic)
    print(f"Delta min: {np.min(Delta)}, Delta max: {np.max(Delta)}")

    # Constraint: exactly k medoids
    bqm.add_linear_equality_constraint([(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=2)

    return bqm




def create_bqm_spread_out(n, k, Delta, importance_values, min_distance, penalty_weight, spread_weight, bias=True):
    """
    Creates a Binary Quadratic Model (BQM) for the k-medoids problem that enforces spread-out selections.

    Parameters:
    - n: Number of data points
    - k: Number of medoids
    - Delta: Similarity (distance) matrix
    - importance_values: Array of importance values for each point
    - min_distance: Minimum allowed distance between medoids in meters
    - penalty_weight: Weight for penalizing close medoids
    - spread_weight: Weight for encouraging spread-out medoid selections
    - bias: Boolean indicating whether to use importance values in the selection

    Returns:
    - bqm: The BQM model
    """
    min_distance_degrees = meters_to_degrees(min_distance)

    bqm = dimod.BinaryQuadraticModel('BINARY')
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    linear = {}
    quadratic = {}

    for i in range(n):
        linear[f'z_{i}'] = 0
        for j in range(n):
            if i != j:
                # Encourage selection of medoids that are farther apart
                if bias:
                    quadratic[(f'z_{i}', f'z_{j}')] = -spread_weight * Delta[i, j] + importance_values[i]
                else:
                    quadratic[(f'z_{i}', f'z_{j}')] = -spread_weight * Delta[i, j]

                # Strong penalty for selecting medoids too close to each other
                if Delta[i, j] < min_distance_degrees:
                    penalty = penalty_weight * (min_distance_degrees - Delta[i, j]) ** 2  # Quadratic penalty
                    quadratic[(f'z_{i}', f'z_{j}')] += penalty

    bqm.add_linear_from(linear)
    bqm.add_quadratic_from(quadratic)

    print(f"Delta min: {np.min(Delta)}, Delta max: {np.max(Delta)}")

    # Constraint: exactly k medoids
    bqm.add_linear_equality_constraint([(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=20)

    return bqm



def create_bqm_min_distance_penalty(n, k, Delta, min_distance, penalty_weight):
    """
    Creates a BQM where a penalty activates if two medoids are selected closer than min_distance.

    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix
    - min_distance: Minimum allowed distance between medoids
    - penalty_weight: Strength of the penalty term

    Returns:
    - bqm: The Binary Quadratic Model
    """

    # Convert min_distance to same units as Delta (e.g., degrees)
    min_distance_degrees = meters_to_degrees(min_distance)
    print(f"Before normalization: {min_distance_degrees}")
    min_distance_degrees= (min_distance_degrees - np.min(Delta)) / (np.max(Delta) - np.min(Delta))
    print(f"After normalization: {min_distance_degrees}")

    
    bqm = dimod.BinaryQuadraticModel('BINARY')

    # Binary variables for medoid selection
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    # Objective function: Encourage medoids to be spread out
    quadratic = {}
    
    for i in range(n):
        for j in range(i + 1, n):  
            if Delta[i, j] < min_distance_degrees:
                penalty = penalty_weight * (min_distance_degrees - Delta[i, j])
                print(f"Adding penalty {penalty} for distance {Delta[i, j]} < {min_distance_degrees}")
                quadratic[(f'z_{i}', f'z_{j}')] = penalty

    # Add quadratic penalties to BQM
    bqm.add_quadratic_from(quadratic)

    # Constraint: Select exactly k medoids
    bqm.add_linear_equality_constraint(
        [(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=penalty_weight
    )

    return bqm



def choose_weights(n, k, Delta, c_p=1.0, c_s=1.0):
    """Automatically chooses penalty and spread weights based on n, k, and distance matrix."""
    avg_distance = np.mean(Delta)  # Average pairwise distance
    penalty_weight = (c_p * avg_distance) / k
    spread_weight = (c_s * avg_distance) / np.sqrt(n)
    return penalty_weight, spread_weight

def create_bqm_even_spread(n, k, Delta, min_distance, lagrange_multiplier, c_p=1.0, c_s=1.0):
    """
    Creates a BQM that ensures an even spatial spread of selected medoids 
    and provides an objective function evaluator.

    Parameters:
    - n: Number of points
    - k: Number of medoids
    - Delta: Distance matrix (n x n)
    - min_distance: Minimum allowed distance between medoids
    - lagrange_multiplier: Strength of the constraint enforcing exactly k medoids
    - c_p: Coefficient for penalty weight selection
    - c_s: Coefficient for spread weight selection

    Returns:
    - bqm: The Binary Quadratic Model
    - compute_objective: A function to evaluate penalties, rewards, and constraints given a solution.
    """
    # Convert min_distance to degrees
    min_distance_degrees = meters_to_degrees(min_distance)

    print(f"Number of Delta less than min_distance: {np.sum(Delta < min_distance_degrees)}")

    # Auto-select weights
    penalty_weight, spread_weight = choose_weights(n, k, Delta, c_p, c_s)

    bqm = dimod.BinaryQuadraticModel('BINARY')
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    # Objective function components
    quadratic = {}

    def compute_objective(selected_medoids, Delta):
        """Computes objective function components for a given set of selected medoids."""
        penalty_term = 0
        spread_term = 0
        constraint_violation = 0

        num_selected = len(selected_medoids)

        # Compute penalties and rewards
        for i in selected_medoids:
            for j in selected_medoids:
                if i < j:
                    distance = Delta[i, j]
                    if distance < min_distance_degrees:
                        penalty_term += penalty_weight * (min_distance_degrees - distance)
                    else:
                        spread_term += -spread_weight * distance
        
        # Compute constraint violation
        constraint_violation = lagrange_multiplier * (num_selected - k) ** 2

        total_objective = penalty_term + spread_term + constraint_violation
        return {
            "total_objective": total_objective,
            "penalty_term": penalty_term,
            "spread_term": spread_term,
            "constraint_violation": constraint_violation
        }

    # Build the BQM
    for i in range(n):
        for j in range(i + 1, n):
            distance = Delta[i, j]
            
            if distance < min_distance_degrees:
                quadratic[(f'z_{i}', f'z_{j}')] = penalty_weight * (min_distance_degrees - distance)
                print(f"Adding penalty {penalty_weight * (min_distance_degrees - distance)} for distance {distance} < {min_distance_degrees}")
            else:
                quadratic[(f'z_{i}', f'z_{j}')] = -spread_weight * distance

    # Add quadratic terms to BQM
    bqm.add_quadratic_from(quadratic)

    # Constraint: Select exactly k medoids
    bqm.add_linear_equality_constraint(
        [(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=lagrange_multiplier
    )

    return bqm, compute_objective













