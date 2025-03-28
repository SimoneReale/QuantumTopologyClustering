import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_charts_from_solutions(folder_path, output_folder):
    """
    Generates charts for clustering quality, coverage, spatial distribution, and demand fairness metrics
    from a folder containing all_solutions JSON files.

    Parameters:
    - folder_path: Path to the folder containing all_solutions JSON files.
    - output_folder: Path to the folder where the charts will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load all solutions from JSON files
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            method = os.path.splitext(file_name)[0]  # Extract method from file name
            with open(os.path.join(folder_path, file_name), "r") as f:
                data = json.load(f)
                for entry in data:
                    entry["method"] = method  # Add method to each entry
                all_data.extend(data)

    # Convert to a DataFrame for easier processing
    df = pd.DataFrame(all_data)

    # Debugging: Print the DataFrame structure
    print("DataFrame Head:")
    print(df.head())
    print("DataFrame Columns:")
    print(df.columns)

    # 1. Clustering Quality Metrics
    # ADSP & MDSP (Box Plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="method", y="avg_distance")
    plt.title("Average Distance to Service Point (ADSP)")
    plt.ylabel("Average Distance")
    plt.xlabel("Method")
    plt.savefig(os.path.join(output_folder, "avg_distance_boxplot.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="method", y="max_distance")
    plt.title("Maximum Distance to Service Point (MDSP)")
    plt.ylabel("Maximum Distance")
    plt.xlabel("Method")
    plt.savefig(os.path.join(output_folder, "max_distance_boxplot.png"))
    plt.close()

    # Standard Deviation of Service Points Distances (Bar Chart)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="method", y="std_distance", ci=None)
    plt.title("Standard Deviation of Service Points Distances")
    plt.ylabel("Standard Deviation")
    plt.xlabel("Method")
    plt.savefig(os.path.join(output_folder, "std_distance_bar_chart.png"))
    plt.close()

    # Min-Max Service Points Distance Ratio (Bar Chart)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="method", y="min_max_ratio", ci=None)
    plt.title("Min-Max Service Points Distance Ratio")
    plt.ylabel("Min-Max Ratio")
    plt.xlabel("Method")
    plt.savefig(os.path.join(output_folder, "min_max_ratio_bar_chart.png"))
    plt.close()

    # 2. Coverage Metrics
    # Percentage of Uncovered Taxi Calls (Stacked Bar Chart)
    df["uncovered_percentage"] = 100 - df["coverage_percentage"]
    coverage_df = df[["method", "coverage_percentage", "uncovered_percentage"]].melt(id_vars="method", 
                                                                                     var_name="Coverage Type", 
                                                                                     value_name="Percentage")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=coverage_df, x="method", y="Percentage", hue="Coverage Type")
    plt.title("Percentage of Covered vs. Uncovered Taxi Calls")
    plt.ylabel("Percentage")
    plt.xlabel("Method")
    plt.savefig(os.path.join(output_folder, "coverage_stacked_bar_chart.png"))
    plt.close()

    # Direct Coverage (Bar Chart)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="method", y="coverage_percentage", ci=None)
    plt.title("Direct Coverage Percentage")
    plt.ylabel("Coverage Percentage")
    plt.xlabel("Method")
    plt.savefig(os.path.join(output_folder, "direct_coverage_bar_chart.png"))
    plt.close()

    # 3. Spatial Distribution Metrics
    # Convex Hull Area (Bar Chart)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="method", y="area_km2", ci=None)
    plt.title("Convex Hull Area")
    plt.ylabel("Area (km²)")
    plt.xlabel("Method")
    plt.savefig(os.path.join(output_folder, "convex_hull_area_bar_chart.png"))
    plt.close()

    # Service Points Density per km² (Scatter Plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="area_km2", y="medoid_density", hue="method", style="method", s=100)
    plt.title("Service Points Density per km² vs Convex Hull Area")
    plt.ylabel("Service Points Density (per km²)")
    plt.xlabel("Convex Hull Area (km²)")
    plt.legend(title="Method")
    plt.savefig(os.path.join(output_folder, "density_scatter_plot.png"))
    plt.close()

    # 4. Demand Fairness
    # Maximum Demand per Service Point (Bar Chart)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="method", y="max_demand_per_medoid", ci=None)
    plt.title("Maximum Demand per Service Point")
    plt.ylabel("Maximum Demand")
    plt.xlabel("Method")
    plt.savefig(os.path.join(output_folder, "max_demand_per_service_point_bar_chart.png"))
    plt.close()

    # Demand-to-Service Point Ratio Standard Deviation (Box Plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="method", y="demand_to_medoid_ratio_std")
    plt.title("Demand-to-Service Point Ratio Standard Deviation")
    plt.ylabel("Standard Deviation")
    plt.xlabel("Method")
    plt.savefig(os.path.join(output_folder, "demand_ratio_std_boxplot.png"))
    plt.close()

    print(f"Charts have been saved to {output_folder}")


if __name__ == "__main__":
    folder_path = "./solutions"
    output_folder = "./plots"
    generate_charts_from_solutions(folder_path, output_folder)