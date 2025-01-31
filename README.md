# Code for Experiments

## Installation (Windows)
Tested on Windows 11 with conda 24.11.2

Create a new conda environment as follows:
```
conda env create --name your_name --file=environment.yml
```
This should install all dependencies to your conda environment path on your machine. 

If not then add to environment.yml
```
prefix: your path to conda...
```
## Installation (Linux)
Tested on Linux Mint 22 kernel 6.8.0-45-generic with conda 24.11.2

Create a new conda environment as follows:
```
conda env create --name your_name --file=linux-environment.yml
```
This should install all dependencies to your conda environment path on your machine. 

If not then add to linux-environment.yml
```
prefix: your path to conda...
```
## Structure
```
📁 Appendix/
│── 📓 Gamma-CI.ipynb - Notebook for Gamma confidence intervals
│── 📓 SWD-Dist.ipynb - GoF for Gamma distribution

📁 ChangeDetection/
│── 📁 datasets/ - Datasets used (HAR,MNISTseq,Occupancy)
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
```R
library(ocp)
library(ecp)
library(jsonlite)
```

## Detection with SWDCP
### Initializing the Detector

To initialize the detector, provide the sequential data, window length, and optional parameters:
```python
from SWCPD import BaseDetector as SWDCP
detector = SWDCP(data, window_length=50, max_history=20, significance=0.05, use_cuda=True)
```
🔹 Parameters:

- data (np.array or torch.Tensor) : input data (obs $\times$ dim)

- window_length (int) : Size of sliding window

- max_history (int, default=20) – Number of past MoM estimates considered for propagating distribution

- significance (float, default=0.05) – Statistical significance level for detecting changes

- use_cuda (bool, default=True) – Use GPU

### Running detector
```python
detector.process_dataloader(n_theta=500, p=2, split=0.5, explanations=False, verbose=True)
detector.evaluate(ground_truth=GroundTruth,tolerance=20)
```
🔹 Parameters:

- n_theta (int, default=500) – Number of Monte Carlo Samples

- p (int, default=2) – Wasserstein order

- split (float, default=0.5) – Ratio for splitting reference and current data windows

- ground_truth (list) - List of true changepoints
