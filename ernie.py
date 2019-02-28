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

from sklearn.cluster import KMeans

class QuantizedLayer(nn.Module):
    def __init__(self, layer, n_clusters, init_method='linear', error_checking=False):
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

        @param layer (nn.Module): Layer to apply quantization to
        @param n_clusters (int): Number of clusters. log_2(n_clusters) is the size in bits of each 
                                 cluster index.
        @param init_method (String): Method to initialize the clusters. Currently only linear init,
                                     the method found to work best for quantization in Han et al. (2015),
                                     is implemented.
        @param error_checking (bool): Flag for verbose K-means and error checking print statements.
        """
        super(QuantizedLayer, self).__init__()
        orig_shape = layer.weight.shape
        layer_weights = layer.weight.detach().flatten().numpy().reshape((-1, 1))
        
        #  Come up with initial centroid locations for the layer weights
        init_centroid_values = self.init_centroid_weights(layer_weights, n_clusters, init_method)
        kmeans = KMeans(n_clusters, init=init_centroid_values, 
                        n_init=1, max_iter=100, precompute_distances=True, 
                        verbose=error_checking)
        kmeans.fit(layer_weights)
        
        # initialize quantized layer as copy of original layer 
        self.q_layer = copy.deepcopy(layer)
        
        centroid_idxs = kmeans.predict(layer_weights)
        self.q_layer.weight = torch.nn.Parameter(torch.tensor(centroid_idxs, dtype=torch.long).view(orig_shape), requires_grad=False)
        centroid_table = torch.tensor(np.array([centroid for centroid in kmeans.cluster_centers_]), dtype=torch.float32)
        self.centroid_table = nn.Embedding.from_pretrained(centroid_table, freeze=False)
        
        if error_checking:
            print("Layer weights: ", layer_weights)
            print("Init centroid values: ", init_centroid_values)
            print("Centroid locations after k means", kmeans.cluster_centers_)
            print("Quantized Layer weights (should be idxs): ", self.q_layer.weight)
            print("Centroid table: ", self.centroid_table)
        
    def init_centroid_weights(self, weights, num_clusters, init_method):
        """
        computes initial centroid locations 
        for instance, min weight, max weight, and then spaced num_centroid apart
        returns centroid mapped to value

        @param weights (ndarray): Array of the weights in the layer to be compressed.
        @param num_clusters (int): Number of clusters (see n_clusters in __init__)
        @param init_method (String): Cluster initialization method (see init_method
                                     in __init__)

        @returns init_centroid_values (ndarray): Initial centroid values for K-means 
                                                 clustering algorithm.
        """
        init_centroid_values = []
        if init_method == 'linear':
            min_weight, max_weight = np.min(weights).item() * 10, np.max(weights).item() * 10
            spacing = (max_weight - min_weight) / (num_clusters + 1)
            init_centroid_values = np.linspace(min_weight + spacing, max_weight - spacing, num_clusters) / 10
        else:
            raise ValueError('Initialize method {} for centroids is unsupported'.format(init_method))
            
        return init_centroid_values.reshape(-1, 1) # reshape for KMeans -- expects centroids, features
        
        
    def forward(self, input_):
        """
        - Somehow replace centroid locations in stored matrix with true centroid weights
        - If that doesn't work, construct PyTorch `Function` https://pytorch.org/docs/master/notes/extending.html
        """
        orig_shape = self.q_layer.weight.shape
        weights = self.centroid_table(self.q_layer.weight.flatten().long()).view(orig_shape)
        out = F.linear(input_, weights, bias=False) # TODO: Quantize bias
        return out
        
def quantize(self, model, dataset, num_centroids):
    """
    1. Iterates through model layers backwards
    
    2. For each layer in the model
    
        2.a Replace the layer with a QuantizedLayer
        
        2.b Train for some number of epochs
    """
        
    pass
        
if __name__=="__main__":
    """ Unit test for quantization
    
    Tests whether quantizing works
    
    Tests whether saving works
    
    """
    linear = nn.Linear(2, 2)
    linear.weight = torch.nn.Parameter(torch.tensor([[0, 0], [1, 3]], dtype=torch.float32))
    input_ = torch.tensor([2, 1], dtype=torch.float32)
    print(input_)
    q_layer = QuantizedLayer(linear, 2, "linear", error_checking=True)
    L1_loss = nn.L1Loss(size_average=False)
    target = torch.tensor([1, -1], dtype=torch.float32)
    out = q_layer(input_)
    loss = sum(target- out)
    print("out: ", out)
    print("loss: ", loss)
    q_layer.zero_grad()
    loss.backward()
    print(q_layer.centroid_table.weight.grad)
    
