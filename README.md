# Code for Experiments

## Installation (Windows)
Create a new conda environment as follows:
```
conda env create --name your_name --file=environment.yml
```
This should install all dependencies to your conda environment path on your machine. If not then add
```
prefix: your path to conda...
```
## Installation (Linux)
Tested on Linux Mint 22 kernel 6.8.0-45-generic with conda 24.11.2

Create a new conda environment as follows:
```
conda env create --name your_name --file=linux-environment.yml
```
This should install all dependencies to your conda environment path on your machine. If not then add
```
prefix: your path to conda...
```
## Structure
```
📁 Appendix/
│── 📓 Gamma-CI.ipynb - Notebook for Gamma confidence intervals
│── 📓 SWD-Dist.ipynb - GoF for Gamma distribution

📁 ChangeDetection/
│── 📁 R/ - E-divisive and BOCPD R-scripts and results
│── 📓 AblationStudy.ipynb - Notebook for ablation studies
│── 📓 HASC.ipynb - CPD analysis on the HASC dataset
│── 📓 KCPA.ipynb - Kernel-based CPD analysis
│── 📓 MNIST.ipynb - CPD on MNIST dataset
│── 📓 Occupancy.ipynb - CPD on occupancy dataset
│── 📝 SWCPD.py - Main CPD algorithm using Sliced Wasserstein Distance
│── 📓 Synthethic.ipynb - Synthetic data experiments
│── 📓 UpperBoundExample.ipynb - Zoomed Plot for CPD method
│── 📝 utilsCPD.py - Utility functions for CPD

📁 Explainability/
│── 📓 Adv-Examples-ART.ipynb - Adversarial examples
│── 📓 MNIST.ipynb - Explainability for MNIST ViT
│── 📄 run_synthethiy.txt - Results for synthetic experiments
│── 📝 run_synthetic.py - Script to run synthetic explainability experiments
│── 📓 Synthetic_Example.ipynb - SWCPD Explainability on synthetic data
│── 📝 utils.py - Utility functions for explainability
```
## Usage of R

Make sure R is installed and install 
```
library(ocp)
library(ecp)
library(jsonlite)
```