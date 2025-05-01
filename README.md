```
 
 ________  _________  ________     
|\   __  \|\___   ___\\   ____\    
\ \  \|\  \|___ \  \_\ \  \___|    
 \ \  \\\  \   \ \  \ \ \  \       
  \ \  \\\  \   \ \  \ \ \  \____  
   \ \_____  \   \ \__\ \ \_______\
    \|___| \__\   \|__|  \|_______|
          \|__|                    
                                   
```

A Python-based repository for clustering edge topologies. This project includes tools for dataset creation, clustering quality evaluation, and visualization of results.

## Features

- **Algorithms and Models**:
  - k-Medoids (native and sklearn implementations)
  - p-Median and hybrid approaches
  - BQM and CQM-based optimization models via our M-DBC clustering approach with simulated and true annealing

- **Dataset Management**:
  - Creation of reduced and filtered datasets
  - Support for 5G antenna and taxi demand data

- **Visualization**:
  - Heatmaps, convex hulls, and clustering quality metrics
  - Distance and coverage analysis

- **Performance Evaluation**:
  - Execution time tracking
  - Metrics for clustering quality, fairness, and spatial distribution

## Repository Structure

- `create_dataset.py`: Tools for generating and preprocessing datasets.
- `test_infrastructure.py`: Framework for testing clustering methods and saving results.
- `draw_plots_test.py`: Visualization and table generation for clustering results.
- `tables/`: Stores generated tables with clustering metrics.
- `plots/`: Contains visualizations of clustering results.
