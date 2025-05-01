```
 
                                                                                                                 
                                                                                                                 
  .g8""8q. `7MMF'   `7MF'    db      `7MN.   `7MF'MMP""MM""YMM `7MMF'   `7MF'`7MMM.     ,MMF'                    
.dP'    `YM. MM       M     ;MM:       MMN.    M  P'   MM   `7   MM       M    MMMb    dPMM                      
dM'      `MM MM       M    ,V^MM.      M YMb   M       MM        MM       M    M YM   ,M MM                      
MM        MM MM       M   ,M  `MM      M  `MN. M       MM        MM       M    M  Mb  M' MM                      
MM.      ,MP MM       M   AbmmmqMA     M   `MM.M       MM        MM       M    M  YM.P'  MM                      
`Mb.    ,dP' YM.     ,M  A'     VML    M     YMM       MM        YM.     ,M    M  `YM'   MM                      
  `"bmmd"'    `bmmmmd"'.AMA.   .AMMA..JML.    YM     .JMML.       `bmmmmd"'  .JML. `'  .JMML.                    
      MMb                                                                                                        
       `bood'                                                                                                    
                                                                                                                 
                                                                                                                 
MMP""MM""YMM   .g8""8q. `7MM"""Mq.   .g8""8q. `7MMF'        .g8""8q.     .g8"""bgd `YMM'   `MM'                  
P'   MM   `7 .dP'    `YM. MM   `MM..dP'    `YM. MM        .dP'    `YM. .dP'     `M   VMA   ,V                    
     MM      dM'      `MM MM   ,M9 dM'      `MM MM        dM'      `MM dM'       `    VMA ,V                     
     MM      MM        MM MMmmdM9  MM        MM MM        MM        MM MM              VMMP                      
     MM      MM.      ,MP MM       MM.      ,MP MM      , MM.      ,MP MM.    `7MMF'    MM                       
     MM      `Mb.    ,dP' MM       `Mb.    ,dP' MM     ,M `Mb.    ,dP' `Mb.     MM      MM                       
   .JMML.      `"bmmd"' .JMML.       `"bmmd"' .JMMmmmmMMM   `"bmmd"'     `"bmmmdPY    .JMML.                     
                                                                                                                 
                                                                                                                 
                                                                                                                 
                                                                                                                 
  .g8"""bgd `7MMF'    `7MMF'   `7MF'.M"""bgd MMP""MM""YMM `7MM"""YMM  `7MM"""Mq.  `7MMF'`7MN.   `7MF' .g8"""bgd  
.dP'     `M   MM        MM       M ,MI    "Y P'   MM   `7   MM    `7    MM   `MM.   MM    MMN.    M .dP'     `M  
dM'       `   MM        MM       M `MMb.          MM        MM   d      MM   ,M9    MM    M YMb   M dM'       `  
MM            MM        MM       M   `YMMNq.      MM        MMmmMM      MMmmdM9     MM    M  `MN. M MM           
MM.           MM      , MM       M .     `MM      MM        MM   Y  ,   MM  YM.     MM    M   `MM.M MM.    `7MMF'
`Mb.     ,'   MM     ,M YM.     ,M Mb     dM      MM        MM     ,M   MM   `Mb.   MM    M     YMM `Mb.     MM  
  `"bmmmd'  .JMMmmmmMMM  `bmmmmd"' P"Ybmmd"     .JMML.    .JMMmmmmMMM .JMML. .JMM..JMML..JML.    YM   `"bmmmdPY  
                                                                                                                 
                                                                                                                 
                   
                                   
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
