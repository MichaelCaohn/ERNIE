'''In this file:
use to test compression/sparsity once we get to that point
feed in original model and compressed model, and compare parameter bits
test pretrained model'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ernie import QuantizedLayer, layer_check, quantize 

#what = torch.load("../pretrained_transformer/transformer-ende-wmt-pyOnmt/sentencepiece.model")
#print(what)

'''can't upload file to github, replace with own pretrained file path
Pretrained model at : http://opennmt.net/Models-py/, under "English->German (WMT)", under the Model column.
The .pt is the state_dict, not sure about .model at the moment.
TODO - can we get original args of the transformer to load the state_dict from .pt'''
pretrained_file_path = "../pretrained_transformer/transformer-ende-wmt-pyOnmt/averaged-10-epoch.pt"

pretrained_model = torch.load(pretrained_file_path) # TODO - specify GPU for non-local run
print(type(pretrained_model))
pretrained_model.eval()

