from otdd import * 
import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from geomloss.sinkhorn_samples import scaling_parameters, sinkhorn_loop, sinkhorn_cost
from geomloss.sinkhorn_samples import softmin_tensorized, log_weights
from geomloss.utils import Sqrt0, sqrt_0, squared_distances, distances

# from geomloss.kernel_samples import kernel_tensorized
# from geomloss.kernel_samples import kernel_tensorized as hausdorff_tensorized

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

class KeopsRoutine():
    
    def routine(self, α, x, β, y, class_xy=None, class_yx=None, class_xx=None, class_yy=None, **kwargs):
        
        raise NotImplementedError

class SinkhornTensorized(KeopsRoutine):
    def __init__(self, cost, debias=True,
                         blur=0.05, reach=None, 
                         diameter=None, scaling=0.5):
        super().__init__()
        
        self.cost = cost
        self.p = cost.p
        self.debias = debias
        self.blur = blur
        self.reach = reach
        self.diameter = diameter
        self.scaling = scaling
        self.eps = self.blur * self.p
        
    def routine(self, α, x, β, y, class_xy=None, class_yx=None, class_xx=None, class_yy=None, **kwargs):
        
        if x.ndim == 2:
            x = x.unsqueeze(0)
        
        if y.ndim == 2:
            y = y.unsqueeze(0)
        
        B, N, D = x.shape
        _, M, _ = y.shape
        
        if self.debias:
            C_xx, C_yy = ( self.cost( x, x.detach()), self.cost( y, y.detach()) )
            
            if class_xx is not None:
                C_xx = (C_xx**2 + class_xx**2)**0.5
                
            if class_yy is not None:
                C_yy = (C_yy**2 + class_yy**2)**0.5


        C_xy, C_yx = ( self.cost( x, y.detach()), self.cost( y, x.detach()) )  # (B,N,M), (B,M,N)
        
        if class_xy is not None:
            C_xy = (C_xy**2 + class_xy**2)**0.5

        if class_yx is not None:
            C_yx = (C_yx**2 + class_yx**2)**0.5


        diameter, ε, ε_s, ρ = scaling_parameters( x, y, self.p, self.blur, 
                                                 self.reach, self.diameter, self.scaling )

        a_x, b_y, a_y, b_x = sinkhorn_loop( softmin_tensorized, 
                                            log_weights(α), log_weights(β), 
                                            C_xx, C_yy, C_xy, C_yx, ε_s, ρ, debias=self.debias )

        F,G = sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x, batch=True, debias=self.debias, potentials=True)
        
        a_i = α.view(-1,1)
        b_i = β.view(1,-1)
        F_i, G_j = F.view(-1,1), G.view(1,-1)
        cost = (F_i + G_j).mean()
        coupling = ((F_i + G_j - C_xy) / self.eps).exp() * (a_i * b_i)
        
        return cost, coupling, C_xy

class SamplesLossTensorized(CostFunction):

    def __init__(self, keops_routine):
        super().__init__(keops_routine.cost, 1000)
        
        self.keops_routine = keops_routine
        
    def get_sample_weights(self, num_samples):
        x_weights = torch.tensor([1/num_samples for i in range(num_samples)])
        return x_weights
    
    def distance(self, x_vals, y_vals, mask_diagonal=False, max_iter=None):
        
        x_weights = self.get_sample_weights(x_vals.shape[0])
        y_weights = self.get_sample_weights(y_vals.shape[0])
        
        x_weights = x_weights.type_as(x_vals)
        y_weights = y_weights.type_as(y_vals)
        
        cost, coupling, C_xy = self.keops_routine.routine(x_weights, x_vals, y_weights, y_vals)
        
        return cost, coupling.squeeze(0), C_xy
        
    def cost_function(self, x_weights, y_weights, M_dists, scale_params):
        
        pass
    
    def label_distances(self, x_vals, y_vals, x_labels, y_labels, gaussian=False):
        
        distances, class_x_dict, class_y_dict = super().label_distances(
                                                            x_vals, y_vals,
                                                            x_labels.cpu().numpy(), y_labels.cpu().numpy(),
                                                            gaussian=gaussian)
        
        return torch.tensor(distances).float(), class_x_dict, class_y_dict
    
    
    def distance_with_labels(self, x_vals, y_vals, x_labels, y_labels,
                             gaussian_class_distance=False, 
                             mask_diagonal=False):
        
        class_vals_xy = self.label_distances(x_vals, y_vals, 
                                          x_labels, y_labels,
                                          gaussian=gaussian_class_distance)
        
        class_vals_xx = self.label_distances(x_vals, x_vals, 
                                          x_labels, x_labels,
                                          gaussian=gaussian_class_distance)
        
        
        class_vals_yy = self.label_distances(y_vals, y_vals, 
                                          y_labels, y_labels,
                                          gaussian=gaussian_class_distance)
        
        d_xy = torch.tensor(get_class_matrix(x_labels, y_labels, *class_vals_xy)).type_as(x_vals)
        
        d_yx = d_xy.T
        
        d_xx = torch.tensor(get_class_matrix(x_labels, x_labels, *class_vals_xx)).type_as(x_vals)
        
        d_yy = torch.tensor(get_class_matrix(y_labels, y_labels, *class_vals_yy)).type_as(x_vals)

        x_weights = self.get_sample_weights(x_vals.shape[0])
        y_weights = self.get_sample_weights(y_vals.shape[0])

        x_weights = x_weights.type_as(x_vals)
        y_weights = y_weights.type_as(y_vals)
        
        cost, coupling, C_xy = self.keops_routine.routine(x_weights, x_vals, y_weights, y_vals,
                                         class_xy=d_xy, class_yx=d_yx, class_xx=d_xx, class_yy=d_yy)
        
        class_distances, class_x_dict, class_y_dict = class_vals_xy

        return cost, coupling, C_xy, class_distances, class_x_dict, class_y_dict
    
  
    def bootstrap_label_distance(self, num_iterations, dataset_x, sample_size_x, 
                                           dataset_y=None, sample_size_y=None, 
                                           min_labelcount=None, gaussian_class_distance=False):

        distances = []
        mask_diagonal=False

        if dataset_y is None:
            dataset_y = dataset_x
            mask_diagonal=True
            
        class_vals_xy = self.label_distances(x_vals, y_vals, 
                                          x_labels, y_labels,
                                          gaussian=gaussian_class_distance)
        
        class_vals_xx = self.label_distances(x_vals, x_vals, 
                                          x_labels, x_labels,
                                          gaussian=gaussian_class_distance)
        
        
        class_vals_yy = self.label_distances(y_vals, y_vals, 
                                          y_labels, y_labels,
                                          gaussian=gaussian_class_distance)
            
        
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

            x_weights = self.get_sample_weights(sample_x.shape[0])
            y_weights = self.get_sample_weights(sample_y.shape[0])

            x_weights = x_weights.type_as(sample_x)
            y_weights = y_weights.type_as(sample_y)
            
            d_xy = torch.tensor(get_class_matrix(label_x, label_y, *class_vals_xy)).type_as(x_vals)

            d_yx = d_xy.T

            d_xx = torch.tensor(get_class_matrix(label_x, label_x, *class_vals_xx)).type_as(x_vals)

            d_yy = torch.tensor(get_class_matrix(label_y, label_y, *class_vals_yy)).type_as(x_vals)

            cost, coupling, C_xy = self.keops_routine.routine(x_weights, sample_x, y_weights, sample_y,
                                         class_xy=d_xy, class_yx=d_yx, class_xx=d_xx, class_yy=d_yy)
            
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
