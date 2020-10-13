import numpy as np
import os
import ot
from collections import Counter
import matplotlib.pyplot as plt
from scipy import linalg
from multiprocessing import Pool

class ArrayDataset():
    '''
    ArrayDataset - base class for array data
    
    Inputs:
        features - ndarray - data array of shape (n_samples x m_features)
        labels - ndaarray, None - data array of shape (n_samples,) or None
    '''
    def __init__(self, features, labels=None):
        self.features = features

        if labels is None:
            labels = np.array([0 for i in range(features.shape[0])])
        self.labels = labels
        self.classes = sorted(list(set(self.labels)))
        
    def __len__(self):
        return self.features.shape[0]
        
    def get_sample_idx(self, sample_size):
        sample_idxs = np.random.choice(np.arange(self.features.shape[0]), sample_size, replace=False)        
        
        return sample_idxs
    
    def sample_with_label(self, sample_size):
        sample_idxs = self.get_sample_idx(sample_size)
        sample = self.features[sample_idxs]
        
        if self.labels is not None:
            sample_labels = self.labels[sample_idxs]
        else:
            sample_labels = np.array([0 for i in range(sample_size)])
            
        return sample, sample_labels
        
    def sample(self, sample_size):
        sample_idxs = self.get_sample_idx(sample_size)
        sample = self.features[sample_idxs]

        return sample

    def subsample(self, total_size, equal_classes=True):

        idxs = np.arange(self.features.shape[0])

        if equal_classes:
            n_samp = total_size//len(self.classes)
            sample_idx = []
            for c in self.classes:
                sample_idx += list(np.random.choice(idxs[self.labels==c],
                                            min(n_samp, (self.labels==c).sum()), replace=False))
        else:
            sample_idx = np.random.choice(idxs,
                                        min(total_size, self.features.shape[0]), replace=False)

        sample_vecs = self.features[sample_idx]
        sample_labels = self.labels[sample_idx]

        return ArrayDataset(sample_vecs, sample_labels)

    
    
class DistanceFunction():
    '''
    DistanceFunction - base distance function class
        
        Methods subclassing DistanceFunction should subclass the __call__ method
        
        __call__ should take in two ndarrays and return a distance matrix
    '''
    def __call__(self, x_vals, y_vals=None, mask_diagonal=False):
        '''
        __call__ - calculates distance between inputs
        
        Inputs:
            x_vals - ndarray - (n_samples, x_features) array
            y_vals - ndarray, None - (m_samples, y_features) array. If None, function calculates self distance
            
        Returns:
            M_dist - ndarray - (n_samples x m_samples) distance matrix
        '''
        pass
    
    def mask_diagonal(self, M_dist):
        '''
        mask_diagonal - sets the diagonal values of a distance matrix to a large number.
                            used for calcualting self-transport distance
                            
        Inputs:
            M_dist - ndarray - (n,m) distance matrix
            
        Returns
            M_dist - ndarray - (n,m) distance matrix
        '''
        fill_val = max(1e6, M_dist.max()*10)
        np.fill_diagonal(M_dist, fill_val)
        return M_dist
    
class POTDistance(DistanceFunction):
    '''
    POTDistance - Distance function for `ot.dist`
    
    Inputs:
        distance_metric - str - distance metric supported by `ot.dist`
                see https://pythonot.github.io/all.html#ot.dist for valid distance metrics
    '''
    def __init__(self, distance_metric='euclidean'):
        super().__init__()
        
        self.distance_metric = distance_metric
        
    def __call__(self, x_vals, y_vals=None, mask_diagonal=False):
        M_dist = ot.dist(x_vals, y_vals, metric=self.distance_metric)
        
        if mask_diagonal:
            M_dist = self.mask_diagonal(M_dist)
            
        return M_dist

class CostFunction():
    '''
    CostFunction - base class for cost function
    
        Methods subclassing CostFunction should subclass the `cost_function` method 
        
        `distance` should take in two ndarrays and return the cost, coupling and distance matrix
        `cost_function` should take in sample weights and a distance matrix and return the cost and coupling matrix
        
    Inputs:
        distance_function - subclass of DistanceFunction
        default_max_iter - int, default max iteration value for cost calculation
    '''
    def __init__(self, distance_function, default_max_iter):
        self.distance_function = distance_function
        self.default_max_iter = default_max_iter
        
    def get_iter(self, max_iter):
        # wrapper to set max_iter if None
        if max_iter is None:
            return self.default_max_iter
        else:
            return max_iter
        
    def get_sample_weights(self, num_samples):
        '''
        get_sample_weights - calculates weight values for input array
            
            Current default is to give uniform weight to all samples
        '''
        x_weights = [1/num_samples for i in range(num_samples)]
        return x_weights
    
    def cost_function(self, x_weights, y_weights, M_dist, max_iter):
        '''
        cost - function to calculate transport cost
        
        Inputs:
            x_weights - ndarray - (n_x,) array of sample weights
            y_weights - ndarray - (n_y,) array of sample weights
            M_dist - ndarray - (n_x, n_y) distance matrix
            max_iter - int - max iterations for cost function
            
        Returns:
            cost - int, ndarray - cost value
            coupling - ndarray, None - coupling matrix if applicable
        '''
        
        raise NotImplementedError
        
    def distance(self, x_vals, y_vals, max_iter=None, mask_diagonal=False):
        
        '''
        distance - function to calculate cost, coupling and distance matrix
        
        Inputs:
            x_vals - ndarray - (x_samples, x_features) ndarray
            y_vals - ndarray - (y_samples, y_features) ndarray
            max_iter - int - max iterations for cost function
            mask_diagonal - bool - if True, distance matrix diagonal is masked
            
        Returns:
            cost - float - transport cost
            coupling - ndarray - (x_samples,y_samples) transport matrix
            M_dist - ndarray - (x_samples,y_samples) distaance matrix
        '''
        M_dist = self.distance_function(x_vals, y_vals, mask_diagonal)
        
        x_weights = self.get_sample_weights(x_vals.shape[0])
        y_weights = self.get_sample_weights(y_vals.shape[0])
        
        cost, coupling = self.cost_function(x_weights, y_weights, M_dist, max_iter)
        
        return cost, coupling, M_dist
    
    def gaussian_distance(self, x_vals, y_vals):
        cost = gaussian_distance(x_vals, y_vals)
        
        return cost
    
    def label_distances(self, x_vals, y_vals, x_labels, y_labels, max_iter=None, gaussian=False):
        '''
        calc_class_distances - calculates a class-based distance matrix
        
        Inputs:
            x_vals - ndarray - (x_samples, x_features) ndarray
            y_vals - ndarray - (y_samples, y_features) ndarray
            x_labels - ndarray - (x_samples,) ndarray
            y_labels - ndarray - (y_samples,) ndarray
            max_iter - int, None - max iterations for `cost_function`. If None, default value in `cost_function` is used
            gaussian - bool - if True, cost is calculated with the Gaussian approximation of the 2-Wasserstein distance 
            
        Returns:
            distances - ndarray - (c_x, c_y) matrix of class-to-class distances
            class_x_dict - dict - dict mapping class labels from `x_labels` to row index values in `distances`
            class_y_dict - dict - dict mapping class labels from `y_labels` to column index values in `distances`
        '''
        
        class_x = sorted(list(set(x_labels)))
        class_y = sorted(list(set(y_labels)))
        
        class_x_dict = {j:i for i,j in enumerate(class_x)}
        class_y_dict = {j:i for i,j in enumerate(class_y)}
        
        distances = np.zeros((len(class_x), len(class_y)))

        if gaussian:
            print('precomputing')
            mu_xs = []
            cov_xs = []

            mu_ys = []
            cov_ys = []

            for c1 in class_x:
                sample_x = x_vals[x_labels==c1]
                mu_xs.append(np.atleast_1d(sample_x.mean(0)))
                cov_xs.append(np.atleast_2d(np.cov(sample_x.T)))

            for c2 in class_y:
                sample_y = y_vals[y_labels==c2]
                mu_ys.append(np.atleast_1d(sample_y.mean(0)))
                cov_ys.append(np.atleast_2d(np.cov(sample_y.T)))
        
        print('computing')
        for i, c1 in enumerate(class_x):
            print(i)
            for j, c2 in enumerate(class_y):
                sample_x = x_vals[x_labels==c1]
                sample_y = y_vals[y_labels==c2]
                
                if gaussian:
                    cost = gaussian_distance_from_stats(mu_xs[i], cov_xs[i], mu_ys[j], cov_ys[j])
                else:
                    cost, coupling, M_dist = self.distance(sample_x, sample_y, max_iter=max_iter)                
                
                distances[i,j] = cost
                
        return distances, class_x_dict, class_y_dict
    
    def distance_with_labels(self, x_vals, y_vals, x_labels, y_labels, max_iter=None,
                             gaussian_class_distance=False, gaussian_data_distance=False, 
                             mask_diagonal=False):
        
        '''
        distance_with_labels - calculates class-based transport distance (OTDD)
        
        Inputs:
            x_vals - ndarray - (x_samples, x_features) ndarray
            y_vals - ndarray - (y_samples, y_features) ndarray
            x_labels - ndarray - (x_samples,) ndarray
            y_labels - ndarray - (y_samples,) ndarray
            max_iter - int, None - max iterations for `cost_function`. If None, 
                                                default value in `cost_function` is used
            gaussian_class_distance - bool - if True, class distances are calculated with the Gaussian 
                                                approximation of the 2-Wasserstein distance 
                                                
            mask_diagonal - bool - if True, distance matrix diagonal is masked (used for intra-distance)
            
        Returns:
            cost - float - transport cost
            coupling - ndarray - (x_samples,y_samples) transport matrix
            OTDD_matrix - ndarray - (x_samples,y_samples) coupling matrix
            class_distances - ndarray - (c_x, c_y) matrix of class-to-class distances
            class_x_dict - dict - dict mapping class labels from `x_labels` to row index values in `distances`
            class_y_dict - dict - dict mapping class labels from `y_labels` to column index values in `distances`
        '''
        
        class_distances, class_x_dict, class_y_dict = self.label_distances(x_vals, y_vals, 
                                                                          x_labels, y_labels, max_iter=max_iter,
                                                                          gaussian=gaussian_class_distance)
        
        if not gaussian_data_distance:
            M_dist = self.distance_function(x_vals, y_vals, mask_diagonal)

            dz = np.zeros(M_dist.shape)

            # TODO: vectorize this
            for i in range(M_dist.shape[0]):
                for j in range(M_dist.shape[1]):
                    c1 = class_x_dict[x_labels[i]]
                    c2 = class_y_dict[y_labels[j]]

                    w_dist = class_distances[c1, c2]
                    dz[i,j] = w_dist

            OTDD_matrix = (M_dist**2 + dz**2)**0.5
        
        else:
            OTDD_matrix = class_distances
        
        if mask_diagonal:
            OTDD_matrix = self.distance_function.mask_diagonal(OTDD_matrix)
        
        cost, coupling = self.cost_function(None, None, OTDD_matrix, max_iter)
        
        return cost, coupling, OTDD_matrix, class_distances, class_x_dict, class_y_dict
    
    def bootstrap_distance(self, num_iterations, dataset_x, sample_size_x, 
                                      dataset_y=None, sample_size_y=None, 
                                     max_iter=None, gaussian=False):
        
        '''
        bootstrap_OT_distance - estimates the value of OT_distance over the entire dataset using bootstrap sampling
        
        Inputs:
            num_iterations - int - number of samples
            dataset_x - ArrayDataset
            sample_size_x - int - sample size for dataset_x
            dataset_y - ArrayDataset, None - if None, function will run intra-sampling on dataset_x
            sample_size_y - int - sample size for dataset_y if not None
        
        Returns:
            distances - ndarray - array of size (num_iterations,) with bootstrap distance values
        '''
                
        distances = []
        
        for i in range(num_iterations):
            
            if dataset_y is not None:
                sample_x = dataset_x.sample(sample_size_x)
                sample_y = dataset_y.sample(sample_size_y)
            
            else:
                sample = dataset_x.sample(sample_size_x*2)
                sample_x = sample[:sample_size_x]
                sample_y = sample[sample_size_x:]
                
            if gaussian:
                cost = self.gaussian_distance(sample_x, sample_y)
            else:
                cost, coupling, M_dist = self.distance(sample_x, sample_y, max_iter=max_iter)
                
            distances.append(cost)
            
        return distances
    
    def bootstrap_label_distance(self, num_iterations, dataset_x, sample_size_x, 
                                           dataset_y=None, sample_size_y=None, 
                                           min_labelcount=None, max_iter=None):
        
        # TODO: bootstrap routine based on class to build class distance matrix proper and avoid instability

        '''
        Notes on improving this. Currently gaussian class distance calculation is slow and unstable for small samples.
        There should be an initial routine that samples from each dataset classwise to build the class distance matrix.
        Then that distance matrix should be applied for all bootstrap samples.
        This could 
        '''

        distances = []
        mask_diagonal=False

        if dataset_y is None:
            dataset_y = dataset_x
            mask_diagonal=True

        class_distances, class_x_dict, class_y_dict = label_distances(dataset_x.features, dataset_y.features, 
                                                    x_labels, y_labels, max_iter=None, gaussian=True)

        for i in range(num_iterations):

            if sample_size_y is not None:
                sample_x, label_x = dataset_x.sample_with_label(sample_size_x)
                sample_y, label_y = dataset_y.sample_with_label(sample_size_y)
            else:
                sample, label = dataset_x.sample_with_label(sample_size_x*2)
                sample_x, label_x = sample[:sample_size_x], label[:sample_size_x]
                sample_y, label_y = sample[sample_size_x:], label[sample_size_x:]

            sample_x, label_x = self.filter_labels(sample_x, label_x, min_labelcount)
            sample_y, label_y = self.filter_labels(sample_y, label_y, min_labelcount)
            
            M_dist = self.distance_function(sample_x, sample_y, mask_diagonal)

            dz = np.zeros(M_dist.shape)

            # TODO: vectorize this
            for i in range(M_dist.shape[0]):
                for j in range(M_dist.shape[1]):
                    c1 = class_x_dict[label_x[i]]
                    c2 = class_y_dict[label_y[j]]

                    w_dist = class_distances[c1, c2]
                    dz[i,j] = w_dist

            OTDD_matrix = (M_dist**2 + dz**2)**0.5

            if mask_diagonal:
                OTDD_matrix = self.distance_function.mask_diagonal(OTDD_matrix)
                
            cost, coupling = self.cost_function(None, None, OTDD_matrix, max_iter)
            
            distances.append(cost)

        return distances

        # distances = []
        
        # for i in range(num_iterations):
            
        #     if dataset_y is not None:
        #         sample_x, label_x = dataset_x.sample_with_label(sample_size_x)
        #         sample_y, label_y = dataset_y.sample_with_label(sample_size_y)
        #     else:
        #         sample, label = dataset_x.sample_with_label(sample_size_x*2)
        #         sample_x, label_x = sample[:sample_size_x], label[:sample_size_x]
        #         sample_y, label_y = sample[sample_size_x:], label[sample_size_x:]
            
        #     sample_x, label_x = self.filter_labels(sample_x, label_x, min_labelcount)
        #     sample_y, label_y = self.filter_labels(sample_y, label_y, min_labelcount)
            
        #     cost, _, _, _, _, _ = self.distance_with_labels(
        #                             sample_x, sample_y, label_x, label_y, 
        #                             max_iter=max_iter, gaussian_class_distance=gaussian_class_distance,
        #                             gaussian_data_distance=gaussian_data_distance)
            
        #     distances.append(cost)
            
        # return distances
    
    def filter_labels(self, x_vals, x_labels, min_labelcount):
        
        if min_labelcount is not None:
            label_counter = Counter(x_labels)

            remove = []
            for label, count in label_counter.items():
                if count < min_labelcount:
                    remove.append(label)

            bools = np.in1d(x_labels, remove)

            x_vals = x_vals[~bools]
            x_labels = x_labels[~bools]
        
        return x_vals, x_labels

    
    
class EarthMoversCost(CostFunction):
    '''
    EarthMoversCost - measures transport cost via Earth Movers distance
    
    Inputs:
        distance_function - subclass of DistanceFunction
        default_max_iter - int, default max iteration value for cost calculation
    '''
    def __init__(self, distance_function, default_max_iter=100000):
        super().__init__(distance_function, default_max_iter)
        
    def cost_function(self, x_weights, y_weights, M_dist, max_iter):
        
        if x_weights is None:
            x_weights = self.get_sample_weights(M_dist.shape[0])
        
        if y_weights is None:
            y_weights = self.get_sample_weights(M_dist.shape[1])
        
        max_iter = self.get_iter(max_iter)
        output = ot.emd2(x_weights, y_weights, M_dist, numItermax=max_iter, return_matrix=True)
    
        cost = output[0]
        coupling = output[1]['G']
        
        return cost, coupling

class SinkhornCost(CostFunction):
    '''
    SinkhornCost - measures transport cost via the entropic regularization optimal transport problem
    
    Inputs:
        distance_function - subclass of DistanceFunction
        default_max_iter - int, default max iteration value for cost calculation
    '''
    def __init__(self, distance_function, entropy, default_max_iter=1000, method='sinkhorn'):
        super().__init__(distance_function, default_max_iter=1000)
        
        self.entropy = entropy
        self.method = 'sinkhorn'
        
    def cost_function(self, x_weights, y_weights, M_dist, max_iter):
        
        if x_weights is None:
            x_weights = self.get_sample_weights(M_dist.shape[0])
        
        if y_weights is None:
            y_weights = self.get_sample_weights(M_dist.shape[1])
        
        max_iter = self.get_iter(max_iter)
        
        output = ot.sinkhorn(x_weights, y_weights, M_dist, self.entropy,
                            method=self.method, numItermax=max_iter, log=True)
        
        coupling = output[0]
        cost = (coupling*M_dist).sum()
        
        return cost, coupling
    

def gaussian_distance_from_stats(mu_x, sigma_x, mu_y, sigma_y, eps=1e-6):
    '''
    gaussian_distance - measures the 2-Wasserstein distance using gaussian approximations,
                            also called the Frechet distance
                            
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        
    Adapted from https://github.com/bioinf-jku/FCD/blob/master/fcd/FCD.py
    
    Inputs
        x_vals - ndarray - (x_samples, x_features) ndarray
        y_vals - ndarray - (y_samples, y_features) ndarray
        
    Returns
        
    '''
    if np.allclose(sigma_x, sigma_y) and np.allclose(mu_x, mu_y):
        cost = 0
    else:

        diff = mu_x - mu_y

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma_x.dot(sigma_y), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_x.shape[0]) * eps
            covmean = linalg.sqrtm((sigma_x + offset).dot(sigma_y + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        cost = (diff.dot(diff) + np.trace(sigma_x) + np.trace(sigma_y) - 2 * tr_covmean)**0.5

    return cost

def gaussian_distance(x_vals, y_vals):
    
    '''
    gaussian_distance - measures the 2-Wasserstein distance using gaussian approximations,
                            also called the Frechet distance
                            
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        
    Adapted from https://github.com/bioinf-jku/FCD/blob/master/fcd/FCD.py
    
    Inputs
        x_vals - ndarray - (x_samples, x_features) ndarray
        y_vals - ndarray - (y_samples, y_features) ndarray
        
    Returns
        
    '''

    mu_x = np.atleast_1d(x_vals.mean(0))
    sigma_x = np.atleast_2d(np.cov(x_vals.T))

    mu_y = np.atleast_1d(y_vals.mean(0))
    sigma_y = np.atleast_2d(np.cov(y_vals.T))

    return gaussian_distance_from_stats(mu_x, sigma_x, mu_y, sigma_y)
    