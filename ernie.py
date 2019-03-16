"""
Class for compression model

Contains function for loading in original model

Contains function for quantizing weights of original model

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from sklearn.cluster import MiniBatchKMeans
from collections import OrderedDict
from onmt.encoders.transformer import TransformerEncoderLayer, TransformerEncoder
from onmt.modules.embeddings import PositionalEncoding

from visualization import compareDistributions, graphCDF

class QuantizedLayer(nn.Module):
    def __init__(self, layer, n_clusters, init_method='linear', error_checking=False, name="", fast=False):
        """
        - Come up with initial centroid locations for the layer weights

        - Run k means w.r.t layer weights
            call scikit learn
            
        - Replace original weights with indices to mapped weights
            
            Options:
                1. construct new layer and then manually iterate through and replace each weight with centroid index
                    - these indices should not be differentiable (check that PyTorch won't do this automatically)
                    - iterate through centroid locatoins and assign weight to closest centroid
            
        - Stored mapped weights / centroid locations
            - these should be differentiable
            - nn.ParameterList() ? 

        Args:
            layer (nn.Module): Layer to apply quantization to
            n_clusters (int): Number of clusters. log_2(n_clusters) is the size in bits of each 
                                 cluster index.
            init_method (String): Method to initialize the clusters. Currently only linear init,
                                     the method found to work best for quantization in Han et al. (2015),
                                     is implemented.
            error_checking (bool): Flag for verbose K-means and error checking print statements.
        """
        super(QuantizedLayer, self).__init__()
        self.pruned = type(layer) == PrunedLayer
        
        self.weight, self.weight_table = self.quantize_params(layer.weight, n_clusters, init_method, error_checking, name, fast)
        
        if layer.bias is not None: # TODO - add check to make sure 2 ** 8 isn't more clusters than layer.bias.numel()
            self.bias, self.bias_table = self.quantize_params(layer.bias, 2 ** 8, init_method, error_checking, name, fast)
        else:
            self.bias = None
    
        
    def init_centroid_weights(self, weights, num_clusters, init_method):
        """ computes initial centroid locations 
        for instance, min weight, max weight, and then spaced num_centroid apart
        returns centroid mapped to value

        Args:
            weights (ndarray): Array of the weights in the layer to be compressed.
            num_clusters (int): Number of clusters (see n_clusters in __init__)
            init_method (String): Cluster initialization method (see init_method
                                     in __init__)
        Returns:
            ndarray: Initial centroid values for K-means clustering algorithm.
        """
        init_centroid_values = []
        if init_method == 'linear':
            min_weight, max_weight = np.min(weights).item() * 10, np.max(weights).item() * 10
            spacing = (max_weight - min_weight) / (num_clusters + 1)
            init_centroid_values = np.linspace(min_weight + spacing, max_weight - spacing, num_clusters) / 10
        else:
            raise ValueError('Initialize method {} for centroids is unsupported'.format(init_method))
        return init_centroid_values.reshape(-1, 1) # reshape for KMeans -- expects centroids, features
        
    def quantize_params(self, params, n_clusters, init_method, error_checking=False, name="", fast=False):
        """ Uses k-means quantization to compress the passed in parameters.

        Args: 
            params (torch.Tensor): tensor of the weights to be quantized
            n_clusters (int): Number of clusters (see n_clusters in __init__)
            init_method (String): Cluster initialization method (see init_method in __init__)
            error_checking (bool): Flag for verbose K-means and error checking print statements.
            fast (bool): 

        Returns:
            (nn.Parameter, nn.Embedding)

            * q_params: The quantized layer weights, which correspond to look up indices for the centroid table.
            * param_table: The centroid table for looking up the weights.
        """
        orig_shape = params.shape
        flat_params = params.detach().flatten().numpy().reshape((-1, 1))
        
        if fast:
            centroid_idxs = [[0] for _ in range(len(flat_params))]
            centroid_table = torch.tensor(np.array([[0] for _ in range(n_clusters)]), dtype=torch.float32)
        else:
            # initialization method supported in scikitlearn KMeans
            if init_method == 'random' or init_method == 'k-means++':
                kmeans = MiniBatchKMeans(n_clusters, init=init_method, n_init=1, max_iter=100, verbose=error_checking)
            # initialization method not in scikitlearn
            else:
                init_centroid_values = self.init_centroid_weights(flat_params, n_clusters, init_method)
                kmeans = MiniBatchKMeans(n_clusters, init=init_centroid_values, n_init=1, max_iter=100, verbose=error_checking)
            kmeans.fit(flat_params)
            centroid_idxs = kmeans.predict(flat_params)
            centroid_table = torch.tensor(np.array([centroid for centroid in kmeans.cluster_centers_]), dtype=torch.float32)
#         np.save("{}_init_cluster_centroids.npy".format(name), kmeans.cluster_centers_)
#         compareDistributions(flat_params, 
#                              np.array(kmeans.cluster_centers_), 
#                              plot_title="{} Centroid Distributions".format(name),
#                              path="distributions/{}_centroids.png".format(name), 
#                              show_fig=True)
        
        q_params = nn.Parameter(torch.tensor(centroid_idxs, dtype=torch.long).view(orig_shape), requires_grad=False)
        param_table = nn.Embedding.from_pretrained(centroid_table, freeze=False)
        
        if error_checking:
            print("Layer weights: ", params)
            print("Init centroid values: ", init_centroid_values)
            print("Centroid locations after k means", kmeans.cluster_centers_)
            print("Quantized Layer weights (should be idxs): ", q_params)
            print("Centroid table: ", param_table)
            
        return q_params, param_table
        
    def forward(self, input_):
        """
        - Somehow replace centroid locations in stored matrix with true centroid weights
        - If that doesn't work, construct PyTorch `Function` https://pytorch.org/docs/master/notes/extending.html
        
        Args:
            input_ (torch.Tensor): Input for the forward pass (x value)

        Returns:
            torch.Tensor: Output of the model after run on the input
        """
        orig_weight_shape, orig_bias_shape = self.weight.shape, self.bias.shape
        weights = self.weight_table(self.weight.flatten().long()).view(orig_weight_shape)
        bias = self.bias_table(self.bias.flatten().long()).view(orig_bias_shape) if self.bias is not None else None
        out = F.linear(input_, weights, bias=bias)
        return out

class BinarizedLayer(nn.Module):

    def __init__(self, layer, n_clusters=2, init_method='linear', error_checking=False, name="", fast=False):
        super(BinarizedLayer, self).__init__()
        self.pruned = False
        if type(layer) == PrunedLayer:
            self.pruned = True
            self.mask = layer.mask
        self.weights = layer.weight
        self.c1, self.c2 = self.binarize_params(layer.weight, init_method, error_checking)
        # TODO - implement bias
        self.bias = layer.bias

    def init_centroid_weights(self, weights, num_clusters, init_method):
        init_centroid_values = []
        if init_method == 'linear':
            min_weight, max_weight = np.min(weights).item() * 10, np.max(weights).item() * 10
            spacing = (max_weight - min_weight) / (num_clusters + 1)
            init_centroid_values = np.linspace(min_weight + spacing, max_weight - spacing, num_clusters) / 10
        else:
            raise ValueError('Initialize method {} for centroids is unsupported'.format(init_method))
        return init_centroid_values.reshape(-1, 1) # reshape for KMeans -- expects centroids, features

    def binarize_params(self, params, init_method, error_checking):
        if self.pruned:
            w = params * self.mask
            w_flat = w.detach().flatten()
            flat_params = w_flat[w_flat.nonzero()].flatten().numpy().reshape((-1, 1))
        else:
            flat_params = params.detach().flatten().numpy().reshape((-1, 1))  

        if init_method == 'random' or init_method == 'k-means++':
            kmeans = MiniBatchKMeans(2, init=init_method, n_init=1, max_iter=100, verbose=error_checking)
        else:
            init_centroid_values = self.init_centroid_weights(flat_params, 2, init_method)
            kmeans = MiniBatchKMeans(2, init=init_centroid_values, n_init=1, max_iter=100, verbose=error_checking)
        kmeans.fit(flat_params)
        c1 = nn.Parameter(torch.tensor(kmeans.cluster_centers_[0], dtype=torch.float32))
        c2 = nn.Parameter(torch.tensor(kmeans.cluster_centers_[1], dtype=torch.float32))
        if error_checking:
            print("Layer weights: ", params)
            print("Initialization method: ", init_method)
            print("Init centroid values: ", init_centroid_values)
            print("Centroids: ", c1, c2)
        return c1, c2

    def forward(self, input_):
        upper = self.c1 if self.c1 >= self.c2 else self.c2
        lower = self.c1 if self.c1 < self.c2 else self.c2
        middle = upper-lower
        w = self.weights.clone()
        if self.pruned:
            w = w * self.mask
            w[(w < middle) & (w != 0)] = lower
            w[(w >= middle) & (w != 0)] = upper
        else:
            #print("orig weights: ", w)
            w[w<middle] = lower
            w[w>=middle] = upper
            #print("new weights: ", w)
        bias = self.bias
        out = F.linear(input_, w, bias=bias)
        return out

class PrunedLayer(nn.Module):
    def __init__(self, layer, prop=0.1):
        """ Implements class-uniform magnitude pruning, ie, prunes x% from the layer passed in
        prop - proportion to prune (eg 0.1 = 10% pruning)
        """
        super(PrunedLayer, self).__init__()
        self.weight = layer.weight
        self.mask = self.prune(layer.weight, prop)
        self.bias = layer.bias

    def prune(self, params, prop):
        # Technically may prune more than prop if there are multiple of the same weight
        k = int(prop*params.numel())
        shape = params.shape
        absv = params.abs()
        tk, idxs = torch.topk(absv.flatten(), k, largest=False, sorted=True)
        threshold = tk[tk.numel()-1].item()
        ones = torch.ones(shape)
        zeros = torch.zeros(shape)
        mask = torch.where(absv > threshold, ones, zeros)
        #indices = [(x // shape[1], x % shape[1]) for x in idxs.tolist()]
        #if self.weight.is_cuda:
        #    return mask.cuda(device=torch.device('cuda', self.weight.get_device()))
        return mask.cuda()
    
    def forward(self, input_):
        '''print("type of weight tensor: ", type(self.weight))
        print("weight tensor on cuda: ", self.weight.is_cuda)
        print("type of mask tensor: ", type(self.mask))
        print("mask tensor on cuda: ", self.mask.is_cuda)
        assert(False)'''
        w = self.weight * self.mask
        bias = self.bias
        out = F.linear(input_, w, bias=bias)
        return out
def layer_check(model, numLin):
    """ Checks that there are no linear layers in the quantized model, and checks that the number of 
    quantized layers is equal to the number of initial linear layers.
    
    Args:
        model (nn.Module): Quantized model
        numLin (int): Number of linear layers in the original model
    """
    numQuant = 0
    numBin = 0
    for l in model.modules():
        if type(l) == nn.Linear:  
            raise ValueError('There should not be any linear layers in a quantized model: {}'.format(model))
        if type(l) == QuantizedLayer:
            numQuant += 1
        if type(l) == BinarizedLayer:
            numBin += 1
    if (numQuant+numBin) != numLin:
        raise ValueError('The number of quantized layers ({}) plus the number of binarized layers ({}) should be equal to the number of linear layers ({})'.format(
            numQuant, numBin, numLin))
        
def quantize(model, num_centroids, error_checking=False, fast=False):
    """
    1. Iterates through model layers forward
    
    2. For each layer in the model
    
        2.a Replace the layer with a QuantizedLayer
    
    Args:
        model (nn.Module): Model to quantize
        num_centroids (int): See n_clusters in QuantizedLayer().__init__()

    Returns:
        nn.Module: model with all layers quantized
    """
    if error_checking:
        num_linear = len([l for l in model.modules() if type(l) == nn.Linear])
        print("original model: ", model)
        print("number of linear layers in original model: ", num_linear) 
        print("=" * 100)
    
    for name, layer in model.named_children():
        if type(layer) == nn.Linear:
            print(name)
            model.__dict__['_modules'][name] = BinarizedLayer(layer, num_centroids, name=name, fast=fast)
        else:
            layer_types = [type(l) for l in layer.modules()]
            if nn.Linear in layer_types:
                quantize(layer, num_centroids, error_checking, fast)
          
    if error_checking:
        layer_check(model, num_linear)
        
    return model

def pruning(model, proportion=0.05, full_model=True):
    for name, layer in model.named_children():
        if type(layer) == nn.Linear:
            print(name)
            model.__dict__['_modules'][name] = PrunedLayer(layer, proportion)
        else:
            layer_types = [type(l) for l in layer.modules()]
            if nn.Linear in layer_types:
                pruning(layer, proportion, full_model=False)
    if full_model:
        print(model)
    return model

if __name__=="__main__":
    """ Unit test for quantization
    
    Tests whether quantizing works
    
    Tests whether saving works
    
    """
    linear = nn.Linear(2, 2, bias=False)
    linear.weight = torch.nn.Parameter(torch.tensor([[0, 0], [1, 3]], dtype=torch.float32))
    #linear.bias = torch.nn.Parameter(torch.tensor([1, 2], dtype=torch.float32))
    input_ = torch.tensor([2, 1], dtype=torch.float32)
    print("=" * 100)
    print("Input: ", input_)
    # commented this out because the q_layer line was yelling at me, can look at later
    b_layer = BinarizedLayer(linear)
    print("centroid 1: ", b_layer.c1)
    print("centroid 2: ", b_layer.c2)
    q_layer = QuantizedLayer(linear, 2)
    print("** quantized centroids ** ", q_layer.weight_table.weight)
    L1_loss = nn.L1Loss(size_average=False)
    target = torch.tensor([-2, -1], dtype=torch.float32)
    out = b_layer(input_)
    loss = sum(target- out)
    print("out: ", out)
    print("loss: ", loss)
    b_layer.zero_grad()
    loss.backward()
    print(b_layer.c1.grad)
    #print(q_layer.c1)
    for f in b_layer.parameters():
        f.data.sub_(f.grad.data * 1)
    #print(q_layer.weight_table.weight)
    print("centroid 1: ", b_layer.c1)
    print("centroid 2: ", b_layer.c2)
    '''out = q_layer(input_)
    loss = sum(target- out)
    print("out: ", out)
    print("loss: ", loss)#'''
    print("quantization test: ")
    print(quantize(nn.Sequential(linear), 2, error_checking=True))
    print("=" * 100)
    bigger_model = nn.Sequential(nn.Linear(3, 3), 
                                nn.Linear(2, 2), 
                                nn.Sequential(nn.Linear(4, 3), nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))))
    
    print("bigger quantization test: ")
    print(quantize(bigger_model, 2, error_checking=True))
    print("=" * 100)
    transf = TransformerEncoder(2, 5, 5, 5, 0, PositionalEncoding(0, 10), 0)
    print("transformer test: ")
    print(transf)

    q_transf = quantize(transf, 2, error_checking=False)
    print(q_transf)
    print("=" * 100)
    print("Pruning test: ")
    linear = nn.Linear(4, 4, bias=False)
    print("Weights of linear layer: ", linear.weight)
    prune = PrunedLayer(linear, 0.25)
    print("Weights of pruned layer: ", prune.weight * prune.mask)
    input_p = torch.tensor([3, 6, 2, 1], dtype=torch.float32)
    target_p = torch.tensor([-3, -6, -2, -1], dtype=torch.float32)
    out_p = prune(input_p)
    loss_p = sum(target_p - out_p)
    print("out: ", out_p)
    print("loss: ", loss_p)
    prune.zero_grad()
    loss_p.backward()
    print(prune.weight.grad)
    print(prune.mask.grad)
    for f in prune.parameters():
        f.data.sub_(f.grad.data * 1)
    print("new weights: ", prune.weight * prune.mask)
    print("new output: ", prune(input_p))
    pq_layer = BinarizedLayer(prune, 2)
    print(pq_layer.pruned)
    print("binarized centroids: ", pq_layer.c1, pq_layer.c2)
    print("=" * 100)
    print("bigger pruning test: ")
    bigger_model = nn.Sequential(nn.Linear(3, 3), 
                                nn.Linear(2, 2), 
                                nn.Sequential(nn.Linear(4, 3), nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))))
    pruned_model = pruning(bigger_model, 0.25)
    print("Pruned model: ", pruned_model)
    for n, l in pruned_model.named_children():
        if type(l) == PrunedLayer:
            print("Weights: ", l.weight * l.mask)
