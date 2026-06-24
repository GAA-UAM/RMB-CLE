# RMB-CLE: Robust Multi-Task Boosting using Clustering and Local Ensembling

This repository contains the official Python implementation of RMB-CLE, a robust multi-task learning framework that mitigates negative transfer through error-driven task clustering and cluster-wise local ensembling.


# Overview

RMB-CLE addresses task heterogeneity in multi-task learning by:

- Estimating task similarity using cross-task generalization errors
- Discovering latent task clusters via hierarchical clustering
- Training cluster-specific local ensembles to enable selective knowledge sharing
- Preventing negative transfer across unrelated tasks

The framework is model-agnostic and is instantiated in this repository using LightGBM and MTGB-based boosting blocks.

## License
The package is licensed under the GNU Lesser General Public [License v2.1](LICENSE).

# Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-repo/rmb-cle.git
cd rmb-cle

python -m venv venv
source venv/bin/activate 

pip install -r requirements.txt
```

# How to Run
To generate toy datasets:
```python
python gen_data.py
```

This creates datasets in:
```python
toy_datasets/{num_clusters}clusters/batch{batch}/
```

## Load Data
Synthetic Data
```python
import pandas as pd

train_df = pd.read_csv("train_reg.csv")

X = train_df.drop(columns=["Target"]).values
y = train_df["Target"].values
```

Real Data
```python
from read_real_data import ReadData

X_train, y_train, X_test, y_test = ReadData(
    dataset="adult_gender",
    random_state=42
)
```

# Train RMB-CLE
```python
from rmb_cle import RMB_CLE
from sklearn.ensemble import HistGradientBoostingRegressor

model = RMB_CLE(
    residual_model_cls=HistGradientBoostingRegressor,
    task_model_cls=HistGradientBoostingRegressor,
    residual_model_as_cls=True,
    n_iter_1st=50,
    n_iter_3rd=50,
    max_iter=100,
    learning_rate=0.1,
    regression=True,
    n_clusters="auto",
    random_state=42
)

model.fit(X, y)
predictions = model.predict(X)
```
Input Format (Important)
```python
[Feature1, Feature2, ..., FeatureD, TaskID]
```
- The last column must be the task identifier.
- Tasks are internally remapped to consecutive indices


## Citations
If you use RMB-CLE in your research or work, please consider citing this project using the following citation format.
```yml
@article{EMAMI2026134327,
title = {Robust multi-task boosting using clustering and local ensembling},
journal = {Neurocomputing},
pages = {134327},
year = {2026},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.134327},
url = {https://www.sciencedirect.com/science/article/pii/S092523122601725X},
author = {Seyedsaman Emami and Daniel HernÃ¡ndez-Lobato and Gonzalo MartÃ­nez-MuÃ±oz}
}
```

## Authors
- [Seyedsaman (Saman) Emami](https://github.com/samanemami/)
- [Daniel Hernández-Lobato](https://github.com/danielhernandezlobato)
- [Gonzalo Martínez-Muñoz](https://github.com/gmarmu)

## Release Information

### Version
0.0.1

### Updated
14 Jan 2026
