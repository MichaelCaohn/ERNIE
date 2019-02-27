"""
Class for compression model

Contains function for loading in original model

Contains function for quantizing weights of original model

"""

import torch
import torch.nn as nn
import numpy as np
import copy

from sklearn.cluster import KMeans

class QuantizedLayer(nn.Module):
    def __init__(self, layer, n_clusters, init_method, error_checking=False):
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
        self.q_layer.weight = torch.nn.Parameter(torch.tensor(centroid_idxs, dtype=torch.uint8).view(orig_shape), requires_grad=False)
        
        self.centroid_table = {i: centroid for i, centroid  in enumerate(kmeans.cluster_centers_)}
        
        if error_checking:
            print("Layer weights flattened: ", layer_weights)
            print("Init centroid values: ", init_centroid_values)
            print("Centroid locations after k means", kmeans.cluster_centers_)
            print("Quantized Layer weights (should be idxs): ", self.q_layer.weight)
            print("Centroid table: ", self.centroid_table)
        
    def init_centroid_weights(self, weights, num_clusters, init_method):
        """
        computes initial centroid locations 
        for instance, min weight, max weight, and then spaced num_centroid apart
        returns centroid mapped to value
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
        raise NotImplementedError()
        
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
    
    q_layer = QuantizedLayer(linear, 2, "linear", error_checking=True)
    print(q_layer.centroid_table)
    
