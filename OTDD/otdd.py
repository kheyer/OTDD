import numpy as np
import os
import ot
import cvxpy as cp
from collections import Counter
import matplotlib.pyplot as plt
from scipy import linalg
from multiprocessing import Pool
import warnings

# Data Classes

class ArrayDataset():
    '''
    ArrayDataset - base class for array data
    
    Inputs:
        features - ndarray - data array of shape (n_samples x m_features)
        labels - ndarray, None - data class label array of shape (n_samples,) or None
        classes - ndarray, None - list that maps integers in `labels` to class names
    '''
    def __init__(self, features, labels=None, classes=None):
        self.features = features

        if labels is None:
            labels = np.array([0 for i in range(features.shape[0])])

        if type(labels[0]) == str:
            labels, class_data = labels_to_ints(labels)
            classes = list(class_data[1].values())

        self.labels = labels

        if classes is None:
            classes = sorted(list(set(self.labels)))

        self.classes = classes
        
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
        classes = list(set(self.labels))

        if equal_classes:
            n_samp = total_size//len(classes)
            sample_idx = []
            for c in classes:
                sample_idx += list(np.random.choice(idxs[self.labels==c],
                                            min(n_samp, (self.labels==c).sum()), replace=False))
        else:
            sample_idx = np.random.choice(idxs,
                                        min(total_size, self.features.shape[0]), replace=False)

        sample_vecs = self.features[sample_idx]
        sample_labels = self.labels[sample_idx]

        return ArrayDataset(sample_vecs, sample_labels, self.classes)

# Distance classes
    
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
        raise NotImplementedError
    
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

    def gaussian_distance(self, x_vals, y_vals):

        raise NotImplementedError

    def gaussian_distance_from_stats(self, mu_x, sigma_x, mu_y, sigma_y, eps=1e-6):

        raise NotImplementedError
    
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

        if not self.distance_metric in ['euclidean', 'sqeuclidean']:
            message = '''
                        Gaussian approximation of the Wasserstein distance not supprted for your \
distance function. Exercise caution in using Gaussian approximations.
                      '''
            warnings.warn(message, Warning)
        
    def __call__(self, x_vals, y_vals=None, mask_diagonal=False):
        M_dist = ot.dist(x_vals, y_vals, metric=self.distance_metric)
        
        if mask_diagonal:
            M_dist = self.mask_diagonal(M_dist)
            
        return M_dist

    def gaussian_distance(self, x_vals, y_vals):
        cost = gaussian_distance(x_vals, y_vals)

        if self.distance_metric == 'euclidean':
            cost = cost**0.5
        
        return cost 

    def gaussian_distance_from_stats(self, mu_x, sigma_x, mu_y, sigma_y, eps=1e-6):

        cost = gaussian_distance_from_stats(mu_x, sigma_x, mu_y, sigma_y, eps=1e-6)

        if self.distance_metric == 'euclidean':
            cost = cost**0.5
        
        return cost 

# def get_class_matrix(x_labels, y_labels, class_distances, class_x_dict, class_y_dict):
    
#     dz = np.zeros((x_labels.shape[0], y_labels.shape[0]))
        
#     for i in range(dz.shape[0]):
#         for j in range(dz.shape[1]):
#             c1 = class_x_dict[x_labels[i].item()]
#             c2 = class_y_dict[y_labels[j].item()]

#             w_dist = class_distances[c1, c2]
#             dz[i,j] = w_dist

#     return dz

def get_class_matrix(x_labels, y_labels, class_distances, class_x_dict=None, class_y_dict=None):
    
    if (class_x_dict is not None) and (not x_labels[0].item() == class_x_dict[x_labels[0].item()]):
        x_labels = np.array([class_x_dict[i.item()] for i in x_labels])
        
    if (class_y_dict is not None) and (not y_labels[0].item() == class_y_dict[y_labels[0].item()]):
        y_labels = np.array([class_y_dict[i.item()] for i in y_labels])
        
    dz = class_distances[x_labels][:, y_labels]
    
    return dz


# Cost Functions

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
            # print(i)
            for j, c2 in enumerate(class_y):
                sample_x = x_vals[x_labels==c1]
                sample_y = y_vals[y_labels==c2]
                
                if gaussian:
                    cost = self.distance_function.gaussian_distance_from_stats(mu_xs[i], cov_xs[i], mu_ys[j], cov_ys[j])
                else:
                    cost, coupling, M_dist = self.distance(sample_x, sample_y, max_iter=max_iter)                
                
                distances[i,j] = cost

        print('distance finished')
                
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

            dz = get_class_matrix(x_labels, y_labels, class_distances, class_x_dict, class_y_dict)

            OTDD_matrix = (M_dist**2 + dz**2)**0.5
        
        else:
            OTDD_matrix = class_distances
        
        if mask_diagonal:
            OTDD_matrix = self.distance_function.mask_diagonal(OTDD_matrix)
        
        print('final cost')
        cost, coupling = self.cost_function(None, None, OTDD_matrix, max_iter)
        
        return cost, coupling, OTDD_matrix, class_distances, class_x_dict, class_y_dict
    
    def bootstrap_distance(self, num_iterations, dataset_x, sample_size_x, 
                                      dataset_y=None, sample_size_y=None, 
                                     max_iter=None, gaussian=False):
        
        '''
        bootstrap_distance - estimates the value of OT_distance over the entire dataset using bootstrap sampling
        
        Inputs:
            num_iterations - int - number of samples
            dataset_x - ArrayDataset
            sample_size_x - int - sample size for dataset_x
            dataset_y - ArrayDataset, None - if None, function will run intra-sampling on dataset_x
            sample_size_y - int - sample size for dataset_y if not None
            max_iter - int, None - max iterations for `cost_function`. If None, 
                                                default value in `cost_function` is used
            gaussian - bool - if True, OT distance will be calculated with Gaussian approximation
        
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
                cost = self.distance_function.gaussian_distance(sample_x, sample_y)
            else:
                cost, coupling, M_dist = self.distance(sample_x, sample_y, max_iter=max_iter)
                
            distances.append(cost)
            
        return distances
    
    def bootstrap_label_distance(self, num_iterations, dataset_x, sample_size_x, 
                                           dataset_y=None, sample_size_y=None, 
                                           min_labelcount=None, max_iter=None):
        '''
        bootstrap_label_distance - estimates the OTDD cost over the entire dataset using bootstrap sampling
        
        Inputs:
            num_iterations - int - number of samples
            dataset_x - ArrayDataset
            sample_size_x - int - sample size for dataset_x
            dataset_y - ArrayDataset, None - if None, function will run intra-sampling on dataset_x
            sample_size_y - int - sample size for dataset_y if not None
            min_labelcount - float, None - minimum class representation in each bootstrap sample (if None will be ignored)
            max_iter - int, None - max iterations for `cost_function`. If None, 
                                                default value in `cost_function` is used

        
        Returns:
            distances - ndarray - array of size (num_iterations,) with bootstrap distance values
        '''

        distances = []
        mask_diagonal=False

        if dataset_y is None:
            dataset_y = dataset_x
            mask_diagonal=True

        class_distances, class_x_dict, class_y_dict = self.label_distances(dataset_x.features, dataset_y.features, 
                                                    dataset_x.labels, dataset_y.labels, 
                                                    max_iter=None, gaussian=True)

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

            dz = get_class_matrix(label_x, label_y, class_distances, class_x_dict, class_y_dict)

            OTDD_matrix = (M_dist**2 + dz**2)**0.5

            if mask_diagonal:
                OTDD_matrix = self.distance_function.mask_diagonal(OTDD_matrix)
                
            cost, coupling = self.cost_function(None, None, OTDD_matrix, max_iter)
            
            distances.append(cost)

        return distances
    
    def filter_labels(self, x_vals, x_labels, min_labelcount):
        '''
        filter_labels - filters `x_vals` and `x_labels` to remove classes that do not have `min_labelcount` members
        '''
        
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
        super().__init__(distance_function, default_max_iter=default_max_iter)
        
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

class CVXPYSolver():
    '''
    CVXPYSolver - Base class for cvxpy solvers designed to give a unified API for solvers with different args
    '''
    def solve(self, objective, constraints, max_iter):
        
        prob = cp.Problem(objective, constraints)
        
        result = self.solve_problem(prob, max_iter)
                
        if (not prob.status == 'optimal') and self.second_solve:
            print("Optimal solution not reached, trying second solve")
            result = self.solve_problem(prob, max_iter)
            
        return result
    
    def solve_problem(self, problem, max_iter):
        raise NotImplementedError
    
        
class SCSSolver(CVXPYSolver):
    '''
    SCSSolver - cvxpy solver using the SCS optimization package
    
    Inputs:
        eps - convergence tolerance
        alpha - relaxation parameter
        scale - balance between minimizing primal and dual residual
        normalize - whether to precondition data matrices
        use_indirect - whether to use indirect solver for KKT sytem (instead of direct)
    '''
    def __init__(self, eps=1e-4, alpha=1.8, scale=5.0, 
                 normalize=True, use_indirect=True, max_iter=1000,
                 second_solve=False):
        super().__init__()
        
        self.solver = 'SCS'
        self.eps = eps
        self.alpha = alpha
        self.scale = scale
        self.normalize = normalize
        self.use_indirect = use_indirect
        self.max_iter = max_iter
        self.second_solve = second_solve
        
    def solve_problem(self, problem, max_iter):
        result = problem.solve(solver=self.solver, max_iters=max_iter, verbose=True,
                               eps=self.eps, alpha=self.alpha, scale=self.scale,
                               normalize=self.normalize, use_indirect=self.use_indirect)
        
        return result
    
    
class ECOSSolver(CVXPYSolver):
    '''
    ECOSSolver - cvxpy solver using the ECOS optimization package
    
    Inputs:
        abstol - absolute accuracy
        reltol - relative accuracy
        feastol - tolerance for feasibility conditions
        abstol_inacc - absolute accuracy for inaccurate solution
        reltol_inacc - relative accuracy for inaccurate solution
        feastol_inacc - tolerance for feasibility condition for inaccurate solution

    '''
    def __init__(self, abstol=1e-8, reltol=1e-8, feastol=1e-8,
                 abstol_inacc=5e-5, reltol_inacc=5e-5, 
                 feastol_inacc=1e-4, max_iter=1000, second_solve=False):
        super().__init__()
        
        self.solver = 'ECOS'
        self.abstol = abstol
        self.reltol = reltol
        self.feastol = feastol
        self.abstol_inacc = abstol_inacc
        self.reltol_inacc = reltol_inacc
        self.feastol_inacc = feastol_inacc
        self.second_solve = second_solve
        self.max_iter = max_iter
        
    def solve_problem(self, problem, max_iter):
        result = problem.solve(solver=self.solver, max_iters=max_iter, verbose=True,
                               abstol=self.abstol, reltol=self.reltol, feastol=self.feastol,
                               abstol_inacc=self.abstol_inacc, reltol_inacc=self.reltol_inacc,
                               feastol_inacc = self.feastol_inacc)
        
        return result

class RobustOTCost(CostFunction):
    '''
    RobustOT - Implements Robust Optimal Transport from arxiv.org/abs/2010.05862
               Implementation based on github.com/yogeshbalaji/robustOT/blob/main/discrete_distributions/solvers/ROT.py
    Inputs:
        distance_function - subclass of DistanceFunction
        rho - optimization constraint described in the paper
        solver - subclass of CVXPYSolver
    '''
    def __init__(self, distance_function, rho, solver):
        super().__init__(distance_function, default_max_iter=solver.max_iter)
    
        self.rho = rho
        self.solver = solver
        
    def cost_function(self, x_weights, y_weights, M_dist, max_iter):
        
        if x_weights is None:
            x_weights = self.get_sample_weights(M_dist.shape[0])
        
        if y_weights is None:
            y_weights = self.get_sample_weights(M_dist.shape[1])
            
        max_iter = self.get_iter(max_iter)
            
        x_weights = np.expand_dims(x_weights, axis=1)
        y_weights = np.expand_dims(y_weights, axis=1)
            
        x_shape, y_shape = M_dist.shape
            
        P = cp.Variable((x_shape, y_shape))
        
        a_tilde = cp.Variable((x_shape, 1))
        b_tilde = cp.Variable((y_shape, 1))
        
        u = np.ones((y_shape, 1))
        v = np.ones((x_shape, 1))
        
        constraints = self.get_constraints(P, u, v, a_tilde, b_tilde, x_weights, y_weights)
        
        objective = cp.Minimize(cp.sum(cp.multiply(P, M_dist)))
        
        result = self.solver.solve(objective, constraints, max_iter)
        
        coupling = np.clip(P.value, 0, float('inf'))
        cost = (coupling * M_dist).sum()
        
        return cost, coupling
    
    def get_constraints(self, P, u, v, a_tilde, b_tilde, x_weights, y_weights):
                
        constraints = [0 <= P, 
                       cp.matmul(P, u) == a_tilde, 
                       cp.matmul(P.T, v) == b_tilde, 
                       0 <= a_tilde, 
                       0 <= b_tilde]
        
        constraints.append(cp.sum([((x_weights[i] - a_tilde[i]) ** 2) / x_weights[i]
                                   for i in range(x_weights.shape[0])]) <= self.rho)
        constraints.append(cp.sum([((y_weights[i] - b_tilde[i]) ** 2) / y_weights[i]
                                   for i in range(y_weights.shape[0])]) <= self.rho)
        
        return constraints
    
    
class ModifiedRobustOTCost(RobustOTCost):
    '''
    ModifiedRobustOTCost - Modified version of RobustOTCost with added constraints for stability
    
    Inputs:
        distance_function - subclass of DistanceFunction
        rho - optimization constraint described in the paper
        solver - cvxpy solving function
        eps - solving tolerance
        second_solve - if True, a second solve step will be attempted if the first fails to converge
    '''
    
    def __init__(self, distance_function, rho, solver):
        super().__init__(distance_function, rho, solver)
        
    def get_constraints(self, P, u, v, a_tilde, b_tilde, x_weights, y_weights):
        constraints = [0 <= P,
                       cp.sum(P) == 1.,
                       cp.matmul(P, u) == a_tilde, 
                       cp.matmul(P.T, v) == b_tilde, 
                       0 <= a_tilde, 
                       0 <= b_tilde]
        
        constraints.append(cp.sum([((x_weights[i] - a_tilde[i]) ** 2) / x_weights[i]
                                   for i in range(x_weights.shape[0])]) <= self.rho)
        constraints.append(cp.sum([((y_weights[i] - b_tilde[i]) ** 2) / y_weights[i]
                                   for i in range(y_weights.shape[0])]) <= self.rho)
        
        return constraints
    

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

        cost = diff.dot(diff) + np.trace(sigma_x) + np.trace(sigma_y) - 2 * tr_covmean

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

def labels_to_ints(labels):
    classes = list(set(labels))
    class_to_id = {classes[i]:i for i in range(len(classes))}
    id_to_class = {i:classes[i] for i in range(len(classes))}

    label_ids = np.array([class_to_id[i] for i in labels])

    return label_ids, [class_to_id, id_to_class]
    