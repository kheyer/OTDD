# Optimal Transport Dataset Distances

This repo is a python implementation of [Geometric Dataset Distances via Optimal Transport](https://arxiv.org/pdf/2002.02923.pdf) and [Robust Optimal Transport](https://arxiv.org/pdf/2010.05862.pdf). Routines are implemented in numpy with [Python Optimal Transport](https://pythonot.github.io/) and [CVXPY](https://www.cvxpy.org/), as well as in Pytorch using [KeOps](https://github.com/getkeops/keops) and [GeomLoss](https://www.kernel-operations.io/geomloss/index.html).

The OTDD algorithm allows us to incorporate label information into the optimal transport problem.

![coupling comparison](https://github.com/kheyer/OTDD/blob/main/media/coupling_comparison.png)

[Algorithm Overview](https://github.com/kheyer/OTDD/tree/main/Algorithms)
[API](https://github.com/kheyer/OTDD/tree/main/API)
[Examples](https://github.com/kheyer/OTDD/tree/main/Examples)

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

## Optimal Transport

Calculating the optimal transport between two sets of feature vectors is done in two steps.
1. Compute pairwise distances between elements in each dataset to generate a cost matrix
2. Use an optimal transport cost function to calculate optimal coupling between data items

The first step is calculated with a `DistanceFunction` method. The second step is calculated with a `CostFunction` method.

ex:

```
distance_function = POTDistance(distance_metric='euclidean')
cost_function = SinkhornCost(distance_function, entropy=0.2)
cost, coupling, M_dist = cost_function.distance(x_vals, y_vals)
```

`cost` is the transport cost between datasets. `coupling` is the coupling matrix solved by the `cost_function` and `M_dist` is the distnce matrix calculated by `distance_function`.


## Optimal Transport Dataset Distances

Traditional optimal transport does not consider the label domain. The main pproch of OTDD is to include the label domain by augmenting the feture distance between two items with the label distance between their respective classes. Label distnces are calculated by computing the optimal transport cost between label subsets of each dataset

<img src="https://render.githubusercontent.com/render/math?math=d_{Z}\bigl((x,y), (x',y') \bigr) \triangleq \bigl( d_{X}(x,x')^p  %2B \text{W}_p^p(\alpha_y, \alpha_{y'}) \bigr)^{\frac{1}{p}}">

To calculate the OTDD distance between two datasets:
1. Compute pairwise distances between elements in each dataset to generate a cost matrix
2. Compute label-to-label optimal transport distances
3. Update the cost matrix with label distances
4. Use an optimal transport cost function to calculate optimal coupling between data items

ex:

```
distance_function = POTDistance(distance_metric='euclidean')
cost_function = SinkhornCost(distance_function, entropy=0.2)
cost, coupling, OTDD_matrix, class_distances, class_x_dict, class_y_dict = cost_function.distance_with_labels(x_vals, y_vals, x_labels, y_label)
```

`cost` is the OTDD transport cost between datasets. `coupling` is the coupling matrix solved by the `cost_function` and `OTDD_matrix` is the distnce matrix calculated by `distance_function` with label-to-label distances. `class_distances` is the matrix of label-to-label distances between datasets. `class_x_dict` and `class_y_dict` map label values in `x_labels` and `y_labels` to index values in `class_distances`

## Gaussian Approximation

For large datasets, computing the optimal transport cost can be prohibative. The transport cost can be approximated by a closed form solution for the 2-Wasserstein distance between two Gaussians, also called the Fréchet distance.

<img src="https://render.githubusercontent.com/render/math?math=\text{W}_2^2(\alpha, \beta) = \| \mu_{\alpha} - \mu_{\beta} \|_2^2 %2B  \| \Sigma^\frac{1}{2}_\alpha - \Sigma_{\beta}^\frac{1}{2} \|_{F}^2">

To calculate the Gaussian approximation distance between two datasets (ie no labels):

```
distance_function = POTDistance(distance_metric='euclidean')
cost_function = SinkhornCost(distance_function, entropy=0.2)

cost = cost_function.gaussian_distance(x_vals, y_vals)
```

To calculate the transport distance with labels, there are two approaches. One is to calculte the label-to-label distances with the Gaussian approximation, then solve the optimal transport coupling between the two datasets

```
distance_function = POTDistance(distance_metric='euclidean')
cost_function = SinkhornCost(distance_function, entropy=0.2)

cost, coupling, OTDD_matrix, class_distances, class_x_dict, class_y_dict = cost_function.distance_with_labels(x_vals, y_vals, x_labels, y_label, gaussian_class_distance=True)
```

An even quicker approach is to calculate the transport cost using the class distance matrix, solving the optimal transport problem over a `c1 x c2` space rather than a `n1 x n2` space.

```
distance_function = POTDistance(distance_metric='euclidean')
cost_function = SinkhornCost(distance_function, entropy=0.2)

cost, coupling, OTDD_matrix, class_distances, class_x_dict, class_y_dict = cost_function.distance_with_labels(x_vals, y_vals, x_labels, y_label, gaussian_class_distance=True, gaussian_data_distance=True)
```

## Bootstap sampling

Another approach to reducing the compute for calculating transport costs is to calculate the transport cost between bootstrapped samples from the data.

For bootstrapping standard transport:

```
distance_function = POTDistance(distance_metric='euclidean')
cost_function = SinkhornCost(distance_function, entropy=0.2)

distances = cost_function.bootstrap_distance(num_iterations, dataset_x, sample_size_x, 
                                      dataset_y, sample_size_y)
```

For bootstrapping transport with labels:

```
distance_function = POTDistance(distance_metric='euclidean')
cost_function = SinkhornCost(distance_function, entropy=0.2)

distances = cost_function.bootstrap_label_distance(num_iterations, dataset_x, sample_size_x, 
                                      dataset_y, sample_size_y)
```

Generally, Gaussian approximations under-estimate the transport cost, while bootstrapping over-estimates the cost.

## Intra-Dataset Distance

To calculate the intra-dataset distance (ie 
<img src="https://render.githubusercontent.com/render/math?math=\text{W}_p^p(\alpha, \alpha)">), pass the `mask_diagonal=True` to the distance methods. The diagonal of the cost matrix (self-distance) will be masked with a large value.

## Plotting

The functions in `plot.py` provide several plotting approaches using [Matplotlib](https://github.com/matplotlib/matplotlib), [HoloViews](https://github.com/holoviz/holoviews) and [Datashader](https://github.com/holoviz/datashader). Datashader methods are recomended for large datasets.

Example with MNIST and USPS digit datasets:

```
mnist = ArrayDataset(mnist_vecs, labels=mnist_labels)
usps = ArrayDataset(usps_vecs, labels=usps_labels)

outputs = cost_fn.distance_with_labels(mnist_sample.features, usps_sample.features,
                                                      mnist_sample.labels, usps_sample.labels,
                                      gaussian_class_distance=True)

cost, coupling, OTDD_matrix, class_distances, class_x_dict, class_y_dict = outputs

emb = TSNE().fit_transform(np.concatenate([mnist.features, usps.features]))
mnist_emb = emb[:mnist.features.shape[0]]
usps_emb = emb[mnist.feaatures.shape[0]:]

```

From here we can plot the coupling matrix

```
plot_coupling(coupling, OTDD_matrix, mnist.labels, usps.labels,
              classes, classes, figsize=(8,8))
```

![coupling plot](https://github.com/kheyer/OTDD/blob/main/media/coupling.png)

A heatmap of class distances

```
plot_class_distances(class_distances, mnist.classes, usps.classes, 
                            cmap='OrRd', figsize=(10,8))
```

![class distance heatmap](https://github.com/kheyer/OTDD/blob/main/media/heatmap.png)

The coupling network based on 2d embeddings

```
plot_coupling_network(mnist_emb, usps_emb, mnist_sample.labels, 
                      usps_sample.labels, coupling, plot_type='hv')
```

<img src="https://github.com/kheyer/OTDD/blob/main/media/connectivity.png" width="500">

If the coupling network is too dense to plot well, we can plot the k strongest connections

```
plot_network_k_connections(mnist_emb, usps_emb, mnist_sample.labels, 
                    usps_sample.labels, coupling, 1000, plot_type='hv')
```

<img src="https://github.com/kheyer/OTDD/blob/main/media/k_connectivity.png" width="500">

