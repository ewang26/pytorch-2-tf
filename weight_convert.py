import argparse
import json
import tqdm
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to .pth file/.pth.tar file')
parser.add_argument('--arch', type=str, help='architecture type of pretrained model')
#TODO support custom architecture definitions

args = parser.parse_args()

path = args.path
arch = args.arch

'''
ensure that all tensor loading is on CPU, as we don't know
what device the original weights were trained on
'''
checkpoint = torch.load(path, map_location = torch.device('cpu'))
print(checkpoint.keys())
import pdb
pdb.set_trace()

def pytorch_weights_to_tf_weights(weight_dict):
    '''
    Converts pytorch weight format to tensorflow weight format, as pytorch
    uses (num, channels, height, width) format while tensorflow uses (num, height, width, channels)
    '''
    tf_weight_dict = {}
    for name, weight in weight_dict.items():
        weight = weight.to('cpu').data.numpy() # convert to numpy array
        if weight.ndim == 4: # check that this is a convolutional layer
            # TODO: Differentiate between conv and depthwise conv
            weight = weight.transpose(2, 3, 0, 1)
            tf_weight_dict[name] = weight
        elif weight.ndim == 2: # this is a linear layer
            weight = weight.transpose(1, 0)
            tf_weight_dict[name] = weight
        else:
            tf_weight_dict[name] = weight
    return weight_dict

def model_arch_conversion():
    model = pretrained



