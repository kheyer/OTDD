from otdd import * 
import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from geomloss.sinkhorn_samples import scaling_parameters, sinkhorn_loop, sinkhorn_cost
from geomloss.sinkhorn_samples import softmin_tensorized, log_weights
from geomloss.utils import Sqrt0, sqrt_0, squared_distances, distances

from geomloss.kernel_samples import kernel_tensorized
from geomloss.kernel_samples import kernel_tensorized as hausdorff_tensorized

class TensorDataset():
    '''
    TensorDataset - base class for tensor data
    
    Inputs:
        features - tensor - data tensor of shape (n_samples x m_features)
        labels - tensor, None - data tensor of shape (n_samples,) or None
    '''
    def __init__(self, features, labels=None):
        self.features = features

        if labels is None:
            labels = torch.tensor([0 for i in range(features.shape[0])])
        self.labels = labels
        self.classes = sorted(list(set(self.labels.numpy())))
        
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
            sample_labels = torch.tensor([0 for i in range(sample_size)])
            
        return sample, sample_labels
        
    def sample(self, sample_size):
        sample_idxs = self.get_sample_idx(sample_size)
        sample = self.features[sample_idxs]

        return sample

    def subsample(self, total_size, equal_classes=True):

        idxs = torch.arange(self.features.shape[0])

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

        return TensorDataset(sample_vecs, sample_labels)


class PytorchDistanceFunction(DistanceFunction):
    '''
    PytorchDistanceFunction - base distance function class for torch tensors
        
        Methods subclassing PytorchDistanceFunction should subclass the __call__ method
        
        __call__ should take in two torch tensors and return a distance matrix
    '''
    def __init__(self):
        super().__init__()
        
    def mask_diagonal(self, M_dist):
        fill_val = max(1e6, M_dist.max()*10)
        mask = torch.eye(M_dist.shape).byte()
        M_dist.masked_fill_(mask, fill_val)
        return M_dist
    
class PytorchEuclideanDistance(PytorchDistanceFunction):
    '''
    PytorchEuclideanDistance - computes tensorized euclidean distance
    '''
    def __init__(self):
        super().__init__()
        self.p = 1.
        
    def __call__(self, x_vals, y_vals=None, mask_diagonal=False):
        
        if y_vals is None:
            M_dist = distances(x_vals, x_vals)
        else:
            M_dist = distances(x_vals, y_vals)
                
        if mask_diagonal:
            M_dist = self.mask_diagonal(M_dist)
            
        return M_dist
    
    def gaussian_distance(self, x_vals, y_vals):
        
        cost = gaussian_distance(x_vals, y_vals)**0.5
        
        return cost 

    def gaussian_distance_from_stats(self, mu_x, sigma_x, mu_y, sigma_y, eps=1e-6):

        cost = gaussian_distance_from_stats(mu_x, sigma_x, mu_y, sigma_y, eps=1e-6)**0.5
        
        return cost 
    
class PytorchEuclideanSquaredDistance(PytorchDistanceFunction):
    '''
    PytorchEuclideanSquaredDistance - computes tensorized squared euclidean distance
    '''
    def __init__(self):
        super().__init__()
        self.p = 2.
        
    def __call__(self, x_vals, y_vals=None, mask_diagonal=False):
        
        if y_vals is None:
            M_dist = squared_distances(x_vals, x_vals)
        else:
            M_dist = squared_distances(x_vals, y_vals)
                
        if mask_diagonal:
            M_dist = self.mask_diagonal(M_dist)
            
        return M_dist
    
    def gaussian_distance(self, x_vals, y_vals):
        
        cost = gaussian_distance(x_vals, y_vals)
        
        return cost 

    def gaussian_distance_from_stats(self, mu_x, sigma_x, mu_y, sigma_y, eps=1e-6):

        cost = gaussian_distance_from_stats(mu_x, sigma_x, mu_y, sigma_y, eps=1e-6)
        
        return cost 

class SamplesLossTensorized(CostFunction):
    '''
    SamplesLossTensorized - measures transport cost via the tensorized sinkhorn distance
    
    Inputs:
        distance_function - subclass of PytorchDistanceFunction
        debias
        blur
        reach
        diameter
        scaling

    '''
    def __init__(self, distance_function, debias=True,
                         blur=0.05, reach=None, 
                         diameter=None, scaling=0.5):
        super().__init__(distance_function, 1000)
        
        self.p = distance_function.p
        self.debias = debias
        self.blur = blur
        self.reach = reach
        self.diameter = diameter
        self.scaling = scaling
        self.eps = self.blur * self.p
        
    def get_sample_weights(self, num_samples):
        x_weights = torch.tensor([1/num_samples for i in range(num_samples)])
        return x_weights
    
    def distance(self, x_vals, y_vals, max_iter=None, mask_diagonal=False):
        
        C_xx = self.distance_function(x_vals, x_vals.detach(), mask_diagonal) if self.debias else None
        C_yy = self.distance_function(y_vals, y_vals.detach(), mask_diagonal) if self.debias else None
        C_xy = self.distance_function(x_vals, y_vals.detach(), mask_diagonal)
        C_yx = self.distance_function(y_vals, x_vals.detach(), mask_diagonal)
        
        M_dists = [C_xx, C_yy, C_xy, C_yx]
        
        scale_params = self.scaling_parameters(x_vals, y_vals)
        
        x_weights = self.get_sample_weights(x_vals.shape[0])
        y_weights = self.get_sample_weights(y_vals.shape[0])
        
        x_weights = x_weights.type_as(x_vals)
        y_weights = y_weights.type_as(y_vals)
        
        cost, coupling = self.cost_function(x_weights, y_weights, M_dists, scale_params)
        
        return cost, coupling.squeeze(0), C_xy
        
    def cost_function(self, x_weights, y_weights, M_dists, scale_params):
        
        C_xx, C_yy, C_xy, C_yx = [i.unsqueeze(0) if i is not None else i for i in M_dists]
        diameter, ε, ε_s, ρ = scale_params
        
        α = x_weights.unsqueeze(0)
        β = y_weights.unsqueeze(0)

        a_x, b_y, a_y, b_x = sinkhorn_loop( softmin_tensorized, 
                                                log_weights(α), log_weights(β), 
                                                C_xx, C_yy, C_xy, C_yx, ε_s, ρ, debias=self.debias)
        
        F, G = sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x, batch=True, debias=self.debias, potentials=True)
        a_i = x_weights.view(-1,1)
        b_i = y_weights.view(1,-1)
        F_i, G_j = F.view(-1,1), G.view(1,-1)
        cost = (F_i + G_j).mean()
        coupling = ((F_i + G_j - C_xy) / self.eps).exp() * (a_i * b_i)
        
        return cost, coupling
        
    def scaling_parameters(self, x_vals, y_vals):
        
        return scaling_parameters(x_vals, y_vals, self.p, self.blur, 
                                  self.reach, self.diameter, self.scaling)
    
    
    
    def label_distances(self, x_vals, y_vals, x_labels, y_labels, max_iter=None, gaussian=False):
        
        class_x = sorted(list(set(x_labels.cpu().numpy())))
        class_y = sorted(list(set(y_labels.cpu().numpy())))
        
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
                    cost = self.distance_function.gaussian_distance_from_stats(mu_xs[i], cov_xs[i], 
                                                                               mu_ys[j], cov_ys[j])
                else:
                    cost, coupling, M_dist = self.distance(sample_x, sample_y, max_iter=max_iter)                
                
                distances[i,j] = cost

        print('distance finished')
                
        return torch.tensor(distances), class_x_dict, class_y_dict
    
    
    def augment_label_distance(self, x_vals, y_vals, x_labels, y_labels, 
                               mask_diagonal=False, gaussian=False,
                               class_vals=None):
        
        C = self.distance_function(x_vals, y_vals, mask_diagonal)
        
        dz = np.zeros(C.shape)
            
        if class_vals is not None:
            class_distances, class_x_dict, class_y_dict = class_vals
            
        else:
            class_distances, class_x_dict, class_y_dict = self.label_distances(x_vals, y_vals, 
                                                          x_labels, y_labels,
                                                          gaussian=gaussian)
        
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                c1 = class_x_dict[x_labels[i].item()]
                c2 = class_y_dict[y_labels[j].item()]

                w_dist = class_distances[c1, c2]
                dz[i,j] = w_dist
                
        dz = torch.tensor(dz)
        O = (C**2 + dz**2)**0.5
        
        return O
        
    
    def distance_with_labels(self, x_vals, y_vals, x_labels, y_labels, max_iter=None,
                             gaussian_class_distance=False, 
                             mask_diagonal=False):
        
        class_vals_xy = self.label_distances(x_vals, y_vals, 
                                          x_labels, y_labels, max_iter=max_iter,
                                          gaussian=gaussian_class_distance)
        
        class_vals_yx = [class_vals_xy[0].T] + list(class_vals_xy[1:])
            
        if self.debias:
            O_xx = self.augment_label_distance(x_vals, x_vals.detach(), 
                                               x_labels, x_labels, 
                                               mask_diagonal, gaussian_class_distance)

            O_yy = self.augment_label_distance(y_vals, y_vals.detach(), 
                                               y_labels, y_labels, 
                                               mask_diagonal, gaussian_class_distance)
        else:
            O_xx = None
            O_yy = None

        O_xy = self.augment_label_distance(x_vals, y_vals.detach(), 
                                           x_labels, y_labels,
                                           mask_diagonal, gaussian_class_distance,
                                           class_vals_xy)

        O_yx = self.augment_label_distance(y_vals, x_vals.detach(),
                                           y_labels, x_labels,
                                           mask_diagonal, gaussian_class_distance,
                                           class_vals_yx)

        M_dists = [O_xx, O_yy, O_xy, O_yx]
        scale_params = self.scaling_parameters(x_vals, y_vals)

        x_weights = self.get_sample_weights(x_vals.shape[0])
        y_weights = self.get_sample_weights(y_vals.shape[0])

        x_weights = x_weights.type_as(x_vals)
        y_weights = y_weights.type_as(y_vals)
        
        cost, coupling = self.cost_function(x_weights, y_weights, M_dists, scale_params)
        
        class_distances, class_x_dict, class_y_dict = class_vals_xy

        return cost, coupling, O_xy, class_distances, class_x_dict, class_y_dict
    
  
    def bootstrap_label_distance(self, num_iterations, dataset_x, sample_size_x, 
                                           dataset_y=None, sample_size_y=None, 
                                           min_labelcount=None, max_iter=None):


        distances = []
        mask_diagonal=False

        if dataset_y is None:
            dataset_y = dataset_x
            mask_diagonal=True
            
        class_vals_xy = self.label_distances(dataset_x.features, dataset_y.features, 
                                          dataset_x.labels, dataset_y.labels, 
                                          gaussian=True)
        
        class_vals_yx = [class_vals_xy[0].T] + list(class_vals_xy[1:])
        
        if self.debias:
        
            class_vals_xx = self.label_distances(dataset_x.features, dataset_x.features, 
                                              dataset_x.labels, dataset_x.labels, 
                                              gaussian=True)

            class_vals_yy = self.label_distances(dataset_y.features, dataset_y.features, 
                                              dataset_y.labels, dataset_y.labels, 
                                              gaussian=True)
        
        
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
            
            
            if self.debias:
                O_xx = self.augment_label_distance(sample_x, sample_x.detach(), 
                                                   label_x, label_x, 
                                                   mask_diagonal, True,
                                                   class_vals_xx)

                O_yy = self.augment_label_distance(sample_y, sample_y.detach(), 
                                                   label_y, label_y, 
                                                   mask_diagonal, True,
                                                   class_vals_yy)
            else:
                O_xx = None
                O_yy = None

            O_xy = self.augment_label_distance(sample_x, sample_y.detach(), 
                                               label_x, label_y,
                                               mask_diagonal, True,
                                               class_vals_xy)

            O_yx = self.augment_label_distance(sample_y, sample_x.detach(),
                                               label_y, label_x,
                                               mask_diagonal, True,
                                               class_vals_yx)
            
            
            M_dists = [O_xx, O_yy, O_xy, O_yx]
            scale_params = self.scaling_parameters(sample_x, sample_y)

            x_weights = self.get_sample_weights(sample_x.shape[0])
            y_weights = self.get_sample_weights(sample_y.shape[0])

            x_weights = x_weights.type_as(sample_x)
            y_weights = y_weights.type_as(sample_y)

            cost, coupling = self.cost_function(x_weights, y_weights, M_dists, scale_params)
            
            distances.append(cost)

        return distances
    
    
    def filter_labels(self, x_vals, x_labels, min_labelcount):
        
        x_labels_np = x_labels.cpu().numpy()

        if min_labelcount is not None:
            label_counter = Counter(x_labels_np)

            remove = []
            for label, count in label_counter.items():
                if count < min_labelcount:
                    remove.append(label)

            bools = np.in1d(x_labels_np, remove)

            x_vals = x_vals[~bools]
            x_labels = x_labels[~bools]
        
        return x_vals, x_labels
    