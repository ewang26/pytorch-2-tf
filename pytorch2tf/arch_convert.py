from pathlib import Path
import argparse
import os
import json
import tqdm
import torch
import torch.nn as nn
import numpy as np

import tensorflow as tf

from models import get_model_arch
from

def model_arch_conversion(arch_string, out_path='~/.pytorch2tf'):
    '''
    Takes in the name of a predefined PyTorch model, extracts the architecture of the model,
    and both dumps the model information in a JSON file defined at out_path/model_arch.json
    and returns a dictionary containing the model information.

    Args:
        arch_string: the name of the model to convert
        out_path: path to write the output JSON file, defaults to ~/.pytorch2tf
    Returns:
        model_arch_dict: a dictionary containing the model information.
    '''
    model_arch = get_model_arch(arch_string)

    model_arch_list = []

    for module in enumerate(model_arch.modules()):
        if isinstance(module, nn.Conv2d):
            model_arch_list.append({
                        'name': module.__class__.__name__,
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding,
                    })
        elif isinstance(module, nn.BatchNorm2d):
            model_arch_list.append({
                        'name': module.__class__.__name__
                    })
        elif isinstance(module, nn.Linear):
            model_arch_list.append({
                        'name': module.__class__.__name__,
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                        'bias': module.bias,
                    })

        # dropout layer            
        elif isinstance(module, nn.Dropout2d):
            model_arch_list.append({
                        'name': module.__class__.__name__,
                        'p': module.p,
                        'inplace': module.inplace,
                    })
        
        # sparse
        elif isinstance(module, nn.Embedding):
            model_arch_list.append({
                        'name': module.__class__.__name__,
                        'num_embeddings': module.num_embeddings,
                        'embedding_dim': module.embedding_dim,
                        'padding_idx': module.padding_idx,
                        'max_norm': module.max_norm,
                        'norm_type': module.norm_type,
                        'scale_grad_by_freq': module.scale_grad_by_freq,
                        'sparse': module.sparse,
                    })
        
        # distance
        elif isinstance(module, nn.CosineSimilarity):
            model_arch_list.append({
                        'name': module.__class__.__name__,
                        'dim': module.dim,
                        'eps': module.eps,
                    })

        else:
            return NotImplementedError('A module within your model is not supported.')

    model_arch_dict = {
            'name': model_arch.__class__.__name__,
            'arch': model_arch_list,
        }

    if out_path == '~/.pytorch2tf': #use the actual relative since '~' doesn't work for all platforms
        out_path = str(Path.home()) + '/.pytorch2tf'

    #create directory if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(out_path, 'model_arch.json'), 'w') as f:
        json.dump(model_arch_dict, f)

    return model_arch_dict

def tf_model_from_dict(in_path, out_path, weight_dict):
    '''
    Takes in a JSON file defining a given model and a dictionary containing
    the weights of the model and generates a TensorFlow model in SavedModel format

    Args:
        in_path: file path to the JSON file defining the model architecture
        out_path: file path to export the converted TF model in SavedModel format
        weight_dict: a dictionary containing the weights for the model

    Returns:
        None
    '''

    assert os.path.exists(in_path), 'path to the model definition does not exist!'

    with open(os.path.join(in_path, 'model_arch.json'), 'r') as f:
        model_arch_dict = json.load(f)

    for i, layer_attr in enumerate(model_arch_dict):
        # get the ith weight in the weight dictionary
        weight = weight_dict[list(weight_dict.keys())[i]]
        layer = create_tf_layer_from_name(layer_attr, weight)

        #TODO add each layer to the TF computational graph or something






