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


## Citations
If you use RMB-CLE in your research or work, please consider citing this project using the following citation format.
```yml

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