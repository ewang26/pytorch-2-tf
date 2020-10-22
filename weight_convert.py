import argparse
import os
import json
import tqdm
import torch
import torch.nn as nn
import numpy as np

from models import get_model_arch
from module_as_dict import module_to_dict

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, help='path to .pth file/.pth.tar file')
parser.add_argument('--out_path', type=str, help='path to output saved model architecture')
parser.add_argument('--arch', type=str, help='architecture type of pretrained model')
#TODO support custom architecture definitions

args = parser.parse_args()

in_path = args.in_path
out_path = args.out_path
arch = args.arch

'''
Ensure that all tensor loading is on CPU, as we don't know
what device the original weights were trained on
'''
#checkpoint = torch.load(in_path, map_location = torch.device('cpu'))

'''
Next, store the model architecture within a JSON file so we can recreate the
architecture in TensorFlow later
'''


def pytorch_weights_to_tf_weights(weight_dict):
    '''
    Converts PyTorch weight format to TensorFlow weight format, as PyTorch
    uses (num, channels, height, width) format while TensorFlow uses (num, height, width, channels)

    Args:
        weight_dict: a dictionary containing the pretrained weights of the PyTorch model

    Returns:
        tf_weight_dict: a dictionary containing the weights in TensorFlow format
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
    return tf_weight_dict

def model_arch_conversion(arch_string, out_path):
    '''
    Takes in the name of a predefined PyTorch model, extracts the architecture of the model, and dumps the model information in a JSON file defined at out_path/model_arch.json

    Args:
        arch_string: the name of the model to convert
        out_path: path to write the output JSON file
    Returns:
        None
    '''
    model_arch = get_model_arch(arch_string)

    model_arch_list = []

    for module in enumerate(model_arch.modules()):
        module_dict = module_to_dict(module)
        if module_dict is None:
            return NotImplementedError('A module within your model is not supported.')

        model_arch_list.append(module_dict)

    model_arch_dict = {
            'name': model_arch.__class__.__name__,
            'arch': model_arch_list,
        }

    print(model_arch_dict)
    with open(os.path.join(out_path, 'model_arch.json'), 'w') as f:
        json.dump(model_arch_dict, f)

model_arch_conversion(arch, out_path)
