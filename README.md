# Code for Experiments

## Installation
Create a new conda environment as follows:
```
conda env create --name your_name --file=environment.yml
```
This should install all dependencies to your conda environment path on your machine. If not then add
```
prefix: your path to conda...
```

## Structure
```
├── Appendix/                     
│   ├── Gamma-CI.ipynb              # Notebook analyzing Gamma confidence intervals
│   ├── SWD-Dist.ipynb              # Notebook on Sliced Wasserstein Distance distribution

├── ChangeDetection/      
    ├── R/                           # E-divisive and BOCPD R-scripts and results        
│   ├── AblationStudy.ipynb          # Notebook for ablation studies
│   ├── HASC.ipynb                   # CPD analysis on the HASC dataset
│   ├── KCPA.ipynb                   # Kernel-based CPD analysis
│   ├── MNIST.ipynb                  # CPD on MNIST dataset
│   ├── Occupancy.ipynb              # CPD on occupancy dataset
│   ├── SWCPD.py                     # Main CPD algorithm using Sliced Wasserstein Distance
│   ├── Synthethic.ipynb             # Synthetic data experiments
│   ├── UpperBoundExample.ipynb      # Example illustrating theoretical upper bounds
│   ├── utilsCPD.py                  # Utility functions for CPD

├── Explainability/                 
│   ├── Adv-Examples-ART.ipynb       # Adversarial examples
│   ├── MNIST.ipynb                  # Explainability for MNIST ViT
│   ├── run_synthethiy.txt           # Results for synthetic experiments
│   ├── run_synthetic.py             # Script to run synthetic explainability experiments
│   ├── Synthetic_Example.ipynb      # Notebook for SWCPD Explainability on snythetic data
│   ├── utils.py                     # Utility functions explainability                  
```

## Usage of R

Make sure R is installed and install 
```
library(ocp)
library(ecp)
library(jsonlite)
```