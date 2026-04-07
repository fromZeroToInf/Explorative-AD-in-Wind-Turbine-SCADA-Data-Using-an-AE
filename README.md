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
It may happen that on MAC OS the internal package importing is corrupted. In this case you need to adjust the imports manually -> check root folder and relative folder expressions.

# Requirements
Make sure you have enough disk space available. 50gb should be enough. 
## Penmanshiel Dataset 
Please visit https://github.com/sltzgs/OpenWindSCADA (recommended) 
or directly https://zenodo.org/records/5946808

# Jupyter Notebooks
You need to run the notebooks in the exact ordering as shown in the main folder.
There is also more content on the folder further_content.
After the last preprocessing step, you can delete the old .csv files and keep only the last versions in order to free disk space.

# Troubleshooting - Replication of the Results
After you completed the preprocessing, please run the notebooks  "further_content/new_pc_filtering.ipynb". It will update the pc filtering.
If the training results differ, you may run torch with cpu instead of gpu. It was not completely possible (at least for me) to get fully deterministic behavior on gpu, it may be possible to re-run the training couple times to get the expected results.
