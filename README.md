# Optimal Transport Dataset Distances

This repo is a python implementation of [Geometric Dataset Distances via Optimal Transport](https://arxiv.org/pdf/2002.02923.pdf) and [Robust Optimal Transport](https://arxiv.org/pdf/2010.05862.pdf). Routines are implemented in numpy with [Python Optimal Transport](https://pythonot.github.io/) and [CVXPY](https://www.cvxpy.org/), as well as in Pytorch using [KeOps](https://github.com/getkeops/keops) and [GeomLoss](https://www.kernel-operations.io/geomloss/index.html).

The OTDD algorithm allows us to incorporate label information into the optimal transport problem.

![coupling comparison](https://github.com/kheyer/OTDD/blob/main/media/coupling_comparison.png)

• [Algorithm Overview](https://github.com/kheyer/OTDD/tree/main/Algorithms)
• [API](https://github.com/kheyer/OTDD/tree/main/API)
• [Examples](https://github.com/kheyer/OTDD/tree/main/Examples)

## Installing

Core dependencies can be installed from the `environment.yml` file

`conda env create -f environment.yml`

To use the Pytorch implementation, install Pytorch, KeOps and GeomLoss

`conda install pytorch torchvision torchaudio -c pytorch`
`pip install pykeops`
`pip install geomloss`

Then validate the KeOps installation

```
import pykeops
pykeops.clean_pykeops()
pykeops.test_torch_bindings() 
```

To use the cheminformatics functions in `chem.py`, install RDKit

`conda install -c rdkit rdkit`
