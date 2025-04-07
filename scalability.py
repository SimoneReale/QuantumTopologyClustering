import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn

# --- Parameters ---
N_csp = 100  # Number of Candidate Service Points (fixed in your experiments)

# Range of Demand Points (DPs) to simulate
# Start from a reasonable number up to a large number typical for edge
M_dp = np.linspace(100, 50000, 100)  # Simulating 100 to 50,000 DPs

# --- Calculate Number of Variables ---
# IBP-M (P-Median based): N (for selecting ASPs) + N*M (for assigning DPs)
variables_ibpm = N_csp + N_csp * M_dp

# M-DBC: N (for selecting ASPs).
# Pre-processing might involve M, but the core QUBO/CQM scales with N^2.
# The number of variables in the optimization itself is N.
variables_mdbc = N_csp * np.ones_like(M_dp)  # Constant N regardless of M

# --- Create Plot ---
sns.set_style("whitegrid")  # Use Seaborn's whitegrid style
plt.figure(figsize=(8, 6))

plt.plot(M_dp, variables_ibpm, label='IBP-M (N + N x M variables)', linewidth=2.5)
plt.plot(M_dp, variables_mdbc, label='M-DBC (N variables)', linestyle='--', linewidth=2.5)

# Use logarithmic scale for the y-axis to show the vast difference
plt.yscale('log')

plt.xlabel('Number of Demand Points (M)', fontsize=14)
plt.ylabel('Number of Optimization Variables (Log Scale)', fontsize=14)
plt.title(f'Scalability Comparison: Optimization Variables vs. Demand Points (N={N_csp} CSPs)', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save or show the plot
plt.savefig('scalability_variables_vs_dp.png', dpi=300)
plt.show()

print("Plot 'scalability_variables_vs_dp.png' generated.")
print(f"Example at M={M_dp[-1]:.0f} DPs:")
print(f"  IBP-M Variables: {variables_ibpm[-1]:.0f}")
print(f"  M-DBC Variables: {variables_mdbc[-1]:.0f}")


