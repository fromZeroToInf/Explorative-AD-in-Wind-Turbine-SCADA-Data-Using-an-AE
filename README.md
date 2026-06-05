# Autoencoder-based Anomaly Detection in Open Wind Turbine SCADA Datasets
This work explores the AE's reconstruction error response to selected synthetic anomalies in a controlled environment. In addition, for anomaly detection, the open-source SCADA Penmanshiel dataset is investigated.

# Installation Guide
Option 1: 
```bash
use pip install -e .
```
Option 2 (recommended): 
Creates own venv...
```bash
python -m pip install pdm
python -m pdm install
```

The package torch needs to be installed manually.

The installation should work on win11 and linux. 
It may happen that on macOS the internal package importing is faulty. In this case you need to adjust the imports manually -> check the root folder and relative folder expressions.

# Requirements
Make sure you have enough disk space available. 50gb should be enough. 
## Acquire the Penmanshiel Dataset and get further Informations 
Please visit https://github.com/sltzgs/OpenWindSCADA (recommended) 
or directly https://zenodo.org/records/5946808

- Instructions for the data are in the notebook step_1_preprocessing.ipynb

# Jupyter Notebooks
You need to run and the notebooks in the exact ordering as shown in the main folder.
There is also more content in the folder further_content.
After the last preprocessing step, you can delete the old .csv files and keep only the last versions in order to free disk space.

# Notes
In the implementation, there exist several functions and code fragments that are not used anymore (dead). 

# Troubleshooting - Replication of Results
- After you complete the preprocessing, you may run the notebook  "further_content/new_pc_filtering.ipynb" in case of inconsistencies in power curve filtering. It will update the power curve filtering.
- It may be possible that the seeding for deterministic behavior on GPU processing is limited. Try to re-run the training a couple times to get the expected results. If it did not succeed, then use the device CPU.

