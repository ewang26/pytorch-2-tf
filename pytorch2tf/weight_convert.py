import os
import json
import tqdm
import torch
import torch.nn as nn
import numpy as np

def pytorch_weights_to_tf_weights(in_path):
    '''
    Converts PyTorch weight format to TensorFlow weight format, as PyTorch
    uses (num, channels, height, width) format while TensorFlow uses (num, height, width, channels)

    Args:
        in_path: the path to the pretrained PyTorch model in .pth/pth.tar format

    Returns:
        tf_weight_dict: a dictionary containing the weights in TensorFlow format
    '''


    '''
    Ensure that all tensor loading is on CPU, as we don't know
    what device the original weights were trained on
    '''
    checkpoint = torch.load(in_path, map_location = torch.device('cpu'))

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






