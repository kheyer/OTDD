from otdd import * 
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from functools import partial
from geomloss.sinkhorn_samples import scaling_parameters, sinkhorn_loop, sinkhorn_cost
from geomloss.sinkhorn_samples import softmin_tensorized, log_weights
from geomloss.utils import Sqrt0, sqrt_0, squared_distances, distances
from pykeops.torch import generic_logsumexp, LazyTensor


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
                                            min(n_samp, (self.labels==c).sum().item()), replace=False))
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

class EuclideanOnline(PytorchEuclideanDistance):
    '''
    EuclideanOnline - computes online euclidean distance
    '''
    def __init__(self):
        super().__init__()
        
        self.formula = "Norm2(X-Y)"
        
    def __call__(self, x_vals, y_vals, mask_diagonal=False, class_values=None):
        
        if type(x_vals) == torch.Tensor:
            x_vals = LazyTensor(x_vals[:,None,:])
            y_vals = LazyTensor(y_vals[None,:,:])

        dist = LazyTensor.norm2(x_vals - y_vals)

        if class_values is not None:
            Z, L = class_values
            
            if type(Z) == torch.Tensor:
                Z = LazyTensor(Z[:,None,:])
                L = LazyTensor(L[None,:,:])

            class_dist = LazyTensor.power(Z | L, 2)

            dist = LazyTensor.sqrt(LazyTensor.power(dist, 2) + class_dist)
        
        return dist
    
class EuclideanSquaredOnline(PytorchEuclideanSquaredDistance):
    '''
    EuclideanSquaredOnline - computes online squared euclidean distance
    '''
    def __init__(self):
        super().__init__()
        
        self.formula = "SqDist(X,Y)"
        
    def __call__(self, x_vals, y_vals, mask_diagonal=False, class_values=None):
        
        if type(x_vals) == torch.Tensor:
            x_vals = LazyTensor(x_vals[:,None,:])
            y_vals = LazyTensor(y_vals[None,:,:])
        
        dist = LazyTensor.norm2(x_vals - y_vals)

        if class_values is not None:
            Z, L = class_values
            
            if type(Z) == torch.Tensor:
                Z = LazyTensor(Z[:,None,:])
                L = LazyTensor(L[None,:,:])

            class_dist = LazyTensor.power(Z | L, 2)

            dist = LazyTensor.sqrt(LazyTensor.power(dist, 2) + class_dist)
        
        return dist


class KeopsRoutine():
    '''
    KeopsRoutine - Base class for implementing keops routines from Geomloss. Designed to work
        with the Geomloss API
    '''
    
    def routine(self, α, x, β, y, class_xy=None, class_yx=None, class_xx=None, class_yy=None, **kwargs):
        
        raise NotImplementedError

class SinkhornTensorized(KeopsRoutine):
    '''
    SinkhornTensorized - Computes tensorized Sinkhorn cost using Keops.
        Best for computing distances with less than 5000 samples

    Inputs:
        cost - Cost function, should be a subclass of PytorchDistanceFunction that returns a dense tensor
        debias - Bool, if True, compute the unbiased Sinkhorn divergence. If False, computee the entropy-regularized “SoftAssign” loss variant
        blur - float, finest level of detail in computing the loss. Prevents overfitting to sample location
        reach - None, float. If float, specifies a maximum scale of interaction between inputs (introduces laziness into the classical Monge problem)
        scaling - float, controlls the trade-off between speed (scaling < .4) and accuracy (scaling > .9)
    '''
    def __init__(self, cost, debias=True,
                         blur=0.05, reach=None, 
                         scaling=0.5):
        super().__init__()
        
        self.cost = cost
        self.p = cost.p
        self.debias = debias
        self.blur = blur
        self.reach = reach
        self.diameter = None
        self.scaling = scaling
        self.eps = self.blur * self.p

    def calculate_cost(self, x, y, class_values):

        C = self.cost(x, y)

        if class_values is not None:
            C = (C**2 + class_values**2)**0.5

        return C
            
    def routine(self, α, x, β, y, class_xy=None, class_yx=None, class_xx=None, class_yy=None, **kwargs):
        
        if x.ndim == 2:
            x = x.unsqueeze(0)
        
        if y.ndim == 2:
            y = y.unsqueeze(0)
        
        B, N, D = x.shape
        _, M, _ = y.shape
        
        if self.debias:
            C_xx = self.calculate_cost(x, x.detach(), class_xx)
            C_yy = self.calculate_cost(y, y.detach(), class_yy)

        else:
            C_xx, C_yy = None, None

        C_xy = self.calculate_cost(x, y.detach(), class_xy) # (B, N, M)
        C_yx = self.calculate_cost(y, x.detach(), class_yx) # (B, M, N)

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

class SinkhornOnline(KeopsRoutine):
    '''
    SinkhornOnline - Computes online Sinkhorn cost using Keops.
        Best for computing distances with more than 5000 samples

    Inputs:
        cost - Cost function, should be a subclass of PytorchDistanceFunction that returns a KeOps LazyTensor
        debias - Bool, if True, compute the unbiased Sinkhorn divergence. If False, computee the entropy-regularized “SoftAssign” loss variant
        blur - float, finest level of detail in computing the loss. Prevents overfitting to sample location
        reach - None, float. If float, specifies a maximum scale of interaction between inputs (introduces laziness into the classical Monge problem)
        scaling - float, controlls the trade-off between speed (scaling < .4) and accuracy (scaling > .9)
        factor_method - str, ['onehot', 'factor']. Specifies how class distances are encoded into KeOps vectors. See SamplesLossOnline
                        docuentation for full details
    '''
    def __init__(self, cost, debias=True,
                         blur=0.05, reach=None, 
                         scaling=0.5, factor_method='onehot'):
        super().__init__()
        
        self.cost = cost
        self.factor_method = factor_method
        self.p = cost.p
        self.debias = debias
        self.blur = blur
        self.reach = reach
        self.diameter = None
        self.scaling = scaling
        self.eps = self.blur * self.p
        
    def logconv(self, C, dtype):
        if len(C) == 2:
            D = C[0].shape[1]
            log_conv = generic_logsumexp("( B - (P * " + self.cost.formula + " ) )",
                                     "A = Vi(1)",
                                     "X = Vi({})".format(D),
                                     "Y = Vj({})".format(D),
                                     "B = Vj(1)",
                                     "P = Pm(1)",
                                     dtype = dtype)
            
        else:
            D = C[0].shape[1]
            C = C[2].shape[1]
            
            log_conv = generic_logsumexp("( B - (P * " + f"Sqrt(Pow({self.cost.formula},2) + Pow(Z|L, 2))" + " ) )",
                                     "A = Vi(1)",
                                     "X = Vi({})".format(D),
                                     "Y = Vj({})".format(D),
                                     "Z = Vi({})".format(C),
                                     "L = Vj({})".format(C),
                                     "B = Vj(1)",
                                     "P = Pm(1)",
                                     dtype = dtype)
            
        return log_conv
        
        
    def softmin_online(self, ε, C_xy, f_y, log_conv=None):
        
        if len(C_xy) == 2:
            x, y = C_xy
            
            return - ε * log_conv( x, y, f_y.view(-1,1), torch.Tensor([1/ε]).type_as(x) ).view(-1)
        
        else:
            x, y, Z, L = C_xy
            
            return - ε * log_conv( x, y, Z, L, f_y.view(-1,1), torch.Tensor([1/ε]).type_as(x) ).view(-1)
        
    def calculate_cost(self, x, y, class_values):
        if class_values is not None:
            if self.factor_method == 'onehot':
                C = (x, y, class_values[0], class_values[1])
            else:
                C = (torch.cat([x, class_values[0]], 1),
                     torch.cat([y, class_values[1]], 1))
        else:
            C = (x, y)
            
        return C
        
        
    def routine(self, α, x, β, y, class_xy=None, class_yx=None, class_xx=None, class_yy=None, **kwargs):
        
        N, D = x.shape
        M, _ = y.shape
        
        if self.debias:
            C_xx = self.calculate_cost(x, x.detach(), class_xx)
            C_yy = self.calculate_cost(y, y.detach(), class_yy)
            
        else:
            C_xx, C_yy = Non
            
        C_xy = self.calculate_cost(x, y.detach(), class_xy)
        C_yx = self.calculate_cost(y, x.detach(), class_yx)
        
        softmin = partial(self.softmin_online, log_conv=self.logconv(C_xy, dtype=str(x.dtype)[6:])) 

        diameter, ε, ε_s, ρ = scaling_parameters( x, y, self.p, self.blur, 
                                                 self.reach, self.diameter, self.scaling )

        a_x, b_y, a_y, b_x = sinkhorn_loop( softmin,
                                            log_weights(α), log_weights(β), 
                                            C_xx, C_yy, C_xy, C_yx, ε_s, ρ, debias=self.debias )

        F,G = sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x, debias=self.debias, potentials=True)
        
        a_i = α.view(-1,1)
        b_i = β.view(1,-1)
        F_i, G_j = F.view(-1,1), G.view(1,-1)
        
        cost = (F_i + G_j).mean()

        # coupling calculation 
        F_i = LazyTensor(F_i.view(-1,1,1))
        G_j = LazyTensor(G_j.view(1,-1,1))
        a_i = LazyTensor(a_i.view(-1,1,1))
        b_j = LazyTensor(b_i.view(1,-1,1))

        if len(C_xy) == 2:
            x,y = C_xy 
            C_ij = self.cost(x,y)
        else:
            x,y,Z,L = C_xy 
            C_ij = self.cost(x,y, class_values=[Z,L])

        coupling = ((F_i + G_j - C_ij) / self.eps).exp() * (a_i * b_j)
        
        return cost, coupling, C_ij


class SamplesLossTensorized(CostFunction):
    '''
    SamplesLossTensorized - cost function for tensorized keops routines
        
    Inputs:
        keops_routine - subclass of KeopsRoutine with a tensorized implementation
    '''

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
        
        return cost, coupling.squeeze(0), C_xy.squeeze(0)
        
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

        return cost, coupling.squeeze(0), C_xy.squeeze(0), class_distances, class_x_dict, class_y_dict
    
  
    def bootstrap_label_distance(self, num_iterations, dataset_x, sample_size_x, 
                                           dataset_y=None, sample_size_y=None, 
                                           min_labelcount=None, gaussian_class_distance=False):

        distances = []
        mask_diagonal=False

        if dataset_y is None:
            dataset_y = dataset_x
            mask_diagonal=True

        x_vals = dataset_x.features
        x_labels = dataset_x.labels
        y_vals = dataset_y.features
        y_labels = dataset_y.labels
            
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

class SamplesLossOnline(SamplesLossTensorized):
    '''
    SamplesLossOnline - cost function for online keops routines
        
    Inputs:
        keops_routine - subclass of KeopsRoutine with a online implementation
        factor_method - str, ['onehot', 'factor']. With online routines, the standard (M,N) matrix of class distances
                            must be encoded on the feature vectors. If 'onehot', class distances are encoded as a matrix
                            of class distances multiplied by a matrix of class one hot encodings. Best when there are fewer classes.
                            If 'factor', class distances are factored into a set of vectors of length `emb_size` and concatenated
                            to feature vectors
        emb_size - int, if factor_method is 'factor', determines the embedding size for class distances
    '''
    def __init__(self, routine, factor_method='onehot', emb_size=5):
        super().__init__(routine)
        self.factor_method = factor_method
        self.keops_routine.factor_method = self.factor_method
        self.emb_size = emb_size
        
    def distance(self, x_vals, y_vals, mask_diagonal=False, max_iter=None):
        
        x_weights = self.get_sample_weights(x_vals.shape[0])
        y_weights = self.get_sample_weights(y_vals.shape[0])
        
        x_weights = x_weights.type_as(x_vals)
        y_weights = y_weights.type_as(y_vals)
        
        cost, coupling, C_xy = self.keops_routine.routine(x_weights, x_vals, y_weights, y_vals)
        
        return cost, coupling, C_xy
    
    def factor_matrix(self, class_distances, x_labels, y_labels):
        if self.factor_method == 'onehot':
            Z = class_distances[x_labels, :]
            L = F.one_hot(y_labels).float()
            output = [Z, L]
            
        else:
            x, y = factor_matrix(class_distances, self.emb_size)
            output = [x[x_labels], y[y_labels]]
            
        return output
    
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
    
        class_xx = self.factor_matrix(class_vals_xx[0], x_labels, x_labels)
        class_yy = self.factor_matrix(class_vals_yy[0], y_labels, y_labels)
        class_xy = self.factor_matrix(class_vals_xy[0], x_labels, y_labels)
        class_yx = [class_xy[1].clone(), class_xy[0].clone()]
        # class_yx = self.factor_matrix(class_vals_xy[0].T, y_labels, x_labels)

        x_weights = self.get_sample_weights(x_vals.shape[0])
        y_weights = self.get_sample_weights(y_vals.shape[0])

        x_weights = x_weights.type_as(x_vals)
        y_weights = y_weights.type_as(y_vals)
        
        cost, coupling, C_xy = self.keops_routine.routine(x_weights, x_vals, y_weights, y_vals,
                                                        class_xy=class_xy, 
                                                        class_yx=class_yx, 
                                                        class_xx=class_xx,
                                                        class_yy=class_yy)
        
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

        x_vals = dataset_x.features 
        x_labels = dataset_x.labels 
        y_vals = dataset_y.features 
        y_labels = dataset_y.labels 
            
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

            class_xx = self.factor_matrix(class_vals_xx[0], label_x, label_x)
            class_yy = self.factor_matrix(class_vals_yy[0], label_y, label_y)
            class_xy = self.factor_matrix(class_vals_xy[0], label_x, label_y)
            class_yx = [class_xy[1].clone(), class_xy[0].clone()]

            cost, coupling, C_xy = self.keops_routine.routine(x_weights, sample_x, y_weights, sample_y,
                                                            class_xy=class_xy, 
                                                            class_yx=class_yx, 
                                                            class_xx=class_xx,
                                                            class_yy=class_yy)
            
            distances.append(cost)

        return distances


def factor_matrix(matrix, emb, lr=1e-1):
    '''
    factor_matrix - factors a (M,N) matrix into vectors of size (M,emb) and (N,emb)

    Inputs:
        matrix - torch.FloatTensor, matrix to be factored
        emb - int, embedding size
        lr - float, learning rate
    '''
    
    x_shape, y_shape = matrix.shape
    
    x = torch.randn((x_shape,emb), requires_grad=True)
    y = torch.randn((y_shape,emb), requires_grad=True)
    
    losses = []

    for i in range(10000):
        dists = (x[:,None,:] - y[None,:,:]).pow(2).sum(-1).pow(0.5)
        loss = (matrix.detach() - dists).pow(2).mean()
        losses.append(loss.item())
        loss.backward()

        x.data.add_(-lr, x.grad.data)
        y.data.add_(-lr, y.grad.data)

        x.grad.detach_()
        x.grad.zero_()

        y.grad.detach_()
        y.grad.zero_()

    if torch.isnan(x).any():
        print('nans')
        x, y = factor_matrix(matrix, emb, lr=lr/5)

    else:
        error = (dists - matrix).abs().mean().item()
        print(f"Factored matrix with {error:.2f} MAE")
    
    return x.detach(), y.detach()
