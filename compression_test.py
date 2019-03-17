'''In this file:
use to test compression/sparsity once we get to that point
feed in original model and compressed model, and compare parameter bits
test pretrained model'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from ernie import PrunedLayer

#what = torch.load("../pretrained_transformer/transformer-ende-wmt-pyOnmt/sentencepiece.model")
#print(what)

'''can't upload file to github, replace with own pretrained file path
Pretrained model at : http://opennmt.net/Models-py/, under "English->German (WMT)", under the Model column.
The .pt is the state_dict, not sure about .model at the moment.
TODO - can we get original args of the transformer to load the state_dict from .pt
pretrained_file_path = "../pretrained_transformer/transformer-ende-wmt-pyOnmt/averaged-10-epoch.pt"

pretrained_model = torch.load(pretrained_file_path) # TODO - specify GPU for non-local run
print(type(pretrained_model))'''

def proportionPruned(model):
	zeros = 0
	ones = 0
	layer = 0
	for module in model.modules():
		if type(module) == PrunedLayer:
			total = module.mask.numel()
			tempOnes = module.mask.nonzero().size(0)
			tempZeros = total - tempOnes
			print("Layer number: ", layer)
			print("Total weights: ", total)
			print("Number of Zeros: ", tempZeros)
			print("Number of Ones: ", tempOnes)
			print("Total Pruning: ", tempZeros/total)
			print("=" * 100)
			layer += 1
			zeros += tempZeros
			ones += tempOnes
	print(model)
	print("Total weights: ", zeros+ones)
	print("Total number of zeros: ", zeros)
	print("Total number of ones: ", ones)
	print("Total Pruning in the Entire Model: ", zeros/(zeros+ones))



