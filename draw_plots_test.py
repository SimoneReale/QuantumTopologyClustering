import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate
import matplotlib.ticker as ticker

def generate_charts_from_solutions(folder_path, output_folder):
    """
    Generates professional charts for clustering quality, coverage, spatial distribution, and demand fairness metrics
    from a folder containing all_solutions JSON files.

    Parameters:
    - folder_path: Path to the folder containing all_solutions JSON files.
    - output_folder: Path to the folder where the charts will be saved.
    """
    # Font size variables
    title_fontsize = 26
    label_fontsize = 24
    tick_fontsize = 22

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

    # Set professional style and context
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)

    # Define a base color palette
    base_palette = sns.color_palette("Set2")

    # Helper function to set y-axis ticks dynamically
    def set_yaxis_ticks(ax, column, data):
        unique_values = sorted(data[column].unique())
        max_ticks = 10  # Maximum number of ticks to display
        if len(unique_values) > max_ticks:
            locator = ticker.MaxNLocator(nbins=max_ticks, prune='both')
            ax.yaxis.set_major_locator(locator)
        else:
            ax.set_yticks(unique_values)

    # 1. Clustering Quality Metrics
    # ADSP & MDSP (Box Plot)
    plt.figure(figsize=(12, 8), dpi=300)
    ax = sns.boxplot(data=df, x="method", y="avg_distance", palette=base_palette[:df["method"].nunique()])
    plt.title("Average Distance to Service Point (ADSP)", fontsize=title_fontsize)
    plt.ylabel("Average Distance (meters)", fontsize=label_fontsize)
    plt.xlabel("Method", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    set_yaxis_ticks(ax, "avg_distance", df)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "avg_distance_boxplot.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8), dpi=300)
    ax = sns.boxplot(data=df, x="method", y="max_distance", palette=base_palette[:df["method"].nunique()])
    plt.title("Maximum Distance to Service Point (MDSP)", fontsize=title_fontsize)
    plt.ylabel("Maximum Distance (meters)", fontsize=label_fontsize)
    plt.xlabel("Method", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    set_yaxis_ticks(ax, "max_distance", df)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "max_distance_boxplot.png"), bbox_inches="tight")
    plt.close()

    # Standard Deviation of Service Points Distances (Bar Chart)
    plt.figure(figsize=(12, 8), dpi=300)
    ax = sns.barplot(data=df, x="method", y="std_distance", ci=None, palette=base_palette[:df["method"].nunique()])
    plt.title("Standard Deviation of Service Points Distances", fontsize=title_fontsize)
    plt.ylabel("Standard Deviation (meters)", fontsize=label_fontsize)
    plt.xlabel("Method", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    set_yaxis_ticks(ax, "std_distance", df)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "std_distance_bar_chart.png"), bbox_inches="tight")
    plt.close()

    # Min-Max Service Points Distance Ratio (Bar Chart)
    plt.figure(figsize=(12, 8), dpi=300)
    ax = sns.barplot(data=df, x="method", y="min_max_ratio", ci=None, palette=base_palette[:df["method"].nunique()])
    plt.title("Min-Max Service Points Distance Ratio", fontsize=title_fontsize)
    plt.ylabel("Min-Max Ratio", fontsize=label_fontsize)
    plt.xlabel("Method", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    set_yaxis_ticks(ax, "min_max_ratio", df)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "min_max_ratio_bar_chart.png"), bbox_inches="tight")
    plt.close()

    # 2. Coverage Metrics
    # Percentage of Uncovered Taxi Calls (Stacked Bar Chart)
    df["uncovered_percentage"] = 100 - df["coverage_percentage"]
    coverage_df = df[["method", "coverage_percentage", "uncovered_percentage"]].melt(
        id_vars="method", var_name="Coverage Type", value_name="Percentage"
    )
    plt.figure(figsize=(12, 8), dpi=300)
    ax = sns.barplot(data=coverage_df, x="method", y="Percentage", hue="Coverage Type", palette=base_palette[:2])
    plt.title("Percentage of Covered vs. Uncovered Taxi Calls", fontsize=title_fontsize)
    plt.ylabel("Percentage", fontsize=label_fontsize)
    plt.xlabel("Method", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    set_yaxis_ticks(ax, "Percentage", coverage_df)
    plt.legend(title="Coverage Type", fontsize=tick_fontsize, title_fontsize=label_fontsize)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "coverage_stacked_bar_chart.png"), bbox_inches="tight")
    plt.close()

    # Direct Coverage (Bar Chart)
    plt.figure(figsize=(12, 8), dpi=300)
    ax = sns.barplot(data=df, x="method", y="coverage_percentage", ci=None, palette=base_palette[:df["method"].nunique()])
    plt.title("Direct Coverage Percentage", fontsize=title_fontsize)
    plt.ylabel("Coverage Percentage", fontsize=label_fontsize)
    plt.xlabel("Method", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    set_yaxis_ticks(ax, "coverage_percentage", df)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "direct_coverage_bar_chart.png"), bbox_inches="tight")
    plt.close()

    # 3. Spatial Distribution Metrics
    # Convex Hull Area (Bar Chart)
    plt.figure(figsize=(12, 8), dpi=300)
    ax = sns.barplot(data=df, x="method", y="area_km2", ci=None, palette=base_palette[:df["method"].nunique()])
    plt.title("Convex Hull Area", fontsize=title_fontsize)
    plt.ylabel("Area (km²)", fontsize=label_fontsize)
    plt.xlabel("Method", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    set_yaxis_ticks(ax, "area_km2", df)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "convex_hull_area_bar_chart.png"), bbox_inches="tight")
    plt.close()

    # Service Points Density per km² (Scatter Plot)
    plt.figure(figsize=(12, 8), dpi=300)
    ax = sns.scatterplot(data=df, x="area_km2", y="medoid_density", hue="method", style="method", s=100, palette=base_palette[:df["method"].nunique()])
    plt.title("Service Points Density per km² vs Convex Hull Area", fontsize=title_fontsize)
    plt.ylabel("Service Points Density (per km²)", fontsize=label_fontsize)
    plt.xlabel("Convex Hull Area (km²)", fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    set_yaxis_ticks(ax, "medoid_density", df)
    plt.legend(title="Method", fontsize=tick_fontsize, title_fontsize=label_fontsize)
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "density_scatter_plot.png"), bbox_inches="tight")
    plt.close()

    # 4. Demand Fairness
    # Maximum Demand per Service Point (Bar Chart)
    plt.figure(figsize=(12, 8), dpi=300)
    ax = sns.barplot(data=df, x="method", y="max_demand_per_medoid", ci=None, palette=base_palette[:df["method"].nunique()])
    plt.title("Maximum Demand per Service Point", fontsize=title_fontsize)
    plt.ylabel("Maximum Demand", fontsize=label_fontsize)
    plt.xlabel("Method", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    set_yaxis_ticks(ax, "max_demand_per_medoid", df)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "max_demand_per_service_point_bar_chart.png"), bbox_inches="tight")
    plt.close()

    # Demand-to-Service Point Ratio Standard Deviation (Box Plot)
    plt.figure(figsize=(12, 8), dpi=300)
    ax = sns.boxplot(data=df, x="method", y="demand_to_medoid_ratio_std", palette=base_palette[:df["method"].nunique()])
    plt.title("Demand-to-Service Point Ratio Standard Deviation", fontsize=title_fontsize)
    plt.ylabel("Standard Deviation", fontsize=label_fontsize)
    plt.xlabel("Method", fontsize=label_fontsize)
    plt.xticks(rotation=45, fontsize=tick_fontsize)
    set_yaxis_ticks(ax, "demand_to_medoid_ratio_std", df)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "demand_ratio_std_boxplot.png"), bbox_inches="tight")
    plt.close()

    print(f"Charts have been saved to {output_folder}")

def generate_tables_from_solutions(folder_path, output_folder):
    """
    Generates tables for clustering quality, coverage, spatial distribution, demand fairness metrics,
    and distance metrics from a folder containing all_solutions JSON files.

    Parameters:
    - folder_path: Path to the folder containing all_solutions JSON files.
    - output_folder: Path to the folder where the tables will be saved.
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
                    # Extract split number from file_name (e.g., "taxi_data/taxi_data_1.txt" -> 1)
                    entry["split_number"] = int(entry["file_name"].split("_")[-1].split(".")[0])
                all_data.extend(data)

    # Convert to a DataFrame for easier processing
    df = pd.DataFrame(all_data)

    # Filter to include only three splits per method
    df = df.groupby("method").apply(lambda x: x.nsmallest(3, "split_number")).reset_index(drop=True)

    # Sort the table by split number
    df = df.sort_values(by=["split_number", "method"]).reset_index(drop=True)

    # 5. Distance Metrics Table
    distance_metrics = []
    for _, row in df.iterrows():
        method = row["method"]
        split_number = row["split_number"]
        all_distances = row.get("all_distances", [])
        area_km2 = row.get("area_km2", None)  # Get the area of the convex hull
        if all_distances:
            min_distance = min(all_distances)
            avg_distance = sum(all_distances) / len(all_distances)
            max_distance = max(all_distances)
            distance_metrics.append({
                "method": method,
                "split_number": split_number,
                "min_distance": min_distance,
                "avg_distance": avg_distance,
                "max_distance": max_distance,
                "area_km2": area_km2  # Include the convex hull area
            })

    # Convert the distance metrics to a DataFrame
    distance_metrics_table = pd.DataFrame(distance_metrics)

    # Save the table ordered by split number
    distance_metrics_table = distance_metrics_table.sort_values(by=["split_number", "method"])
    distance_metrics_table.to_csv(os.path.join(output_folder, "distance_metrics.csv"), index=False)
    print("Distance Metrics table saved.")

    print(f"Tables have been saved to {output_folder}")

def visualize_tables(tables_folder):
    """
    Visualizes the tables generated by `generate_tables_from_solutions`.

    Parameters:
    - tables_folder: Path to the folder containing the CSV tables.
    """
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(tables_folder) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Display each table
    for csv_file in csv_files:
        file_path = os.path.join(tables_folder, csv_file)
        print(f"\nTable: {csv_file}")
        print("=" * (len(csv_file) + 7))
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Display the table using tabulate
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

if __name__ == "__main__":
    folder_path = "./solutions"
    output_folder_plots = "./plots"
    generate_charts_from_solutions(folder_path, output_folder_plots)
    output_folder_tables = "./tables"
    generate_tables_from_solutions(folder_path, output_folder_tables)
    tables_folder = "./tables"  # Path to the folder containing the tables
    visualize_tables(tables_folder)