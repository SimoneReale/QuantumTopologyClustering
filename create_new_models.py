import dimod
import numpy as np
from create_dataset import create_pechino_dataset
import matplotlib.pyplot as plt


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
    # normalizzo min_distance tra 0 e 1
    min_distance_normalized = min_distance / Delta.max()

    Delta = Delta / Delta.max()


    # Auto-select weights
    penalty_weight, spread_weight = choose_weights(n, k, Delta, c_p, c_s)

    bqm = dimod.BinaryQuadraticModel('BINARY')
    z = {i: dimod.Binary(f'z_{i}') for i in range(n)}

    # Objective function components
    quadratic = {}

    def compute_objective(selected_medoids, Delta):
        """Computes objective function components for a given set of selected medoids."""
        #normalizzo Delta tra 0 e 1
        Delta = Delta / Delta.max()
        penalty_term = 0
        spread_term = 0
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
            #print(f"distance {distance} less than min_distance_normalized {min_distance_normalized}: {distance < min_distance_normalized}")
            if distance < min_distance_normalized:
                quadratic[(f'z_{i}', f'z_{j}')] = penalty_weight * (min_distance_normalized - distance)
                #print(f"Adding penalty {penalty_weight * (min_distance_normalized - distance)} for distance {distance} < {min_distance_normalized}")
            else:
                #print(f"Adding bonus for distance {distance} > {min_distance_normalized} equal to {-spread_weight * distance}")
                quadratic[(f'z_{i}', f'z_{j}')] = -spread_weight * distance

    # Add quadratic terms to BQM
    bqm.add_quadratic_from(quadratic)

    # Constraint: Select exactly k medoids
    bqm.add_linear_equality_constraint(
        [(f'z_{i}', 1) for i in range(n)], constant=-k, lagrange_multiplier=lagrange_multiplier
    )

    return bqm, compute_objective

def plot_distance_spread(Delta, importance_values_standard, importance_values_radius):
    """
    Plots the spread of distances in the Delta matrix and the distributions of importance values.

    Parameters:
    - Delta: Distance matrix
    - importance_values_standard: Importance values calculated using the standard method
    - importance_values_radius: Importance values calculated using the radius method
    """
    distances = Delta[np.triu_indices_from(Delta, k=1)]  # Extract upper triangular part of the matrix

    plt.figure(figsize=(15, 10))

    # Plot the spread of distances
    plt.subplot(2, 1, 1)
    plt.hist(distances, bins=50, edgecolor='black')
    plt.title('Spread of Distances in Delta Matrix')
    plt.xlabel('Distance (meters)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Plot the distributions of importance values
    plt.subplot(2, 1, 2)
    plt.hist(importance_values_standard, bins=50, alpha=0.5, label='Standard Method', edgecolor='black')
    plt.hist(importance_values_radius, bins=50, alpha=0.5, label='Radius Method', edgecolor='black')
    plt.title('Distributions of Importance Values')
    plt.xlabel('Importance Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Delta, n, df_5g, df_taxi, importance_values = create_pechino_dataset()
    create_bqm_even_spread(n, 20, Delta, 300, 1, 10000, 1)