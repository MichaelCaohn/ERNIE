"""
Class for compression model

Contains function for loading in original model

Contains function for quantizing weights of original model

"""

import torch
import torch.nn as nn

class QuantizedLayer(nn.Module):
    def __init__(self, layer, num_centroids, init_method):
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
        #  Come up with initial centroid locations for the layer weights
        centroid_values = self.init_centroid_weights(layer.weight, num_centroids, init_method)
        
        
        
    def init_centroid_weights(weights, num_clusters, init_method):
        """
        computes initial centroid locations 
        for instance, min weight, max weight, and then spaced num_centroid apart
        returns centroid mapped to value
        """
        centroid_values = []
        if init_method == 'linear':
            min_weight, max_weight = torch.min(weights), torch.max(weights)
            for centroid_value in range(min_weight, max_weight, num_clusters):
                centroid_values.append(centroid_value)
        else:
            raise ValueError('Initialize method {} for centroids is unsupported'.format(init_method))
            
        return centroid_values
        
        
    def forward(self, input_):
        """
        - Somehow replace centroid locations in stored matrix with true centroid weights
        - If that doesn't work, construct PyTorch `Function` https://pytorch.org/docs/master/notes/extending.html
        """
        
def quantize(self, model, dataset, num_centroids):
    """
    1. Iterates through model layers backwards
    
    2. For each layer in the model
    
        2.a Replace the layer with a QuantizedLayer
        
        2.b Train for some number of epochs
    """
        

        
if __name__=="__main__":
    """ Unit test for quantization
    
    Tests whether quantizing works
    
    Tests whether saving works
    
    """
    linear = nn.Linear(10)
    
