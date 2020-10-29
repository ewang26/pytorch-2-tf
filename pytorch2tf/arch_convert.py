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
from models import create_tf_layer_from_name

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


class TFfromTorchNet:
    '''
    Wrapper class to construct a TensorFlow model from the saved PyTorch weights and a JSON file describing the model configuration.

    Attributes:
        graph: the computational graph of the model
        input_shape: the shape of the input tensors the model accepts
        n_classes: number of output classes for the model
    '''

    def __init__(self, path, input_shape=None, output_classes=100):
        self.graph = tf.Graph()
        self.input_shape = input_shape
        self.n_classes = output_classes

        self._path = path
        assert os.path.exists(path), 'path to the model definition does not exist!'
        if self._path.endswith('json'):
            self.net_config = json.load(open(self._path, 'r'))
            self._path = '/'.join(self._path.split('/')[:-1])
        else:
            self.net_config = json.load(open(os.path.join(self._path, 'model_arch.json'), 'r'))

        self._logs_path, self._save_path = None, None

        with self.graph.as_default():
            self._define_inputs()
            logits = self.build()
            with tf.variable_scope('L2_Loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

            prediction = logits
            # losses
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
            )
            self.cross_entropy = cross_entropy

            correct_prediction = tf.equal(
                tf.argmax(prediction, 1),
                tf.argmax(self.labels, 1)
            )
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # optimizer and train step
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9, use_nesterov=True)
            self.train_step = optimizer.minimize(cross_entropy + l2_loss * 4e-5)

            self.global_variables_initializer = tf.global_variables_initializer()
            self._count_trainable_params()
            self.saver = tf.train.Saver()
        self._initialize_session()

    def save_path(self):
        '''
        Initializes the save_path to save the model to
        Args:
            None
        Returns:
            None
        '''
        if self._save_path is None:
            save_path = '%s/checkpoint' % self._path
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return self._save_path

    def logs_path(self):
        '''
        Initializes the path to save logging files
        Args:
            None
        Returns:
            None
        '''
        if self._logs_path is None:
            logs_path = '%s/logs' % self._path
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    def _initialize_session(self):
        '''
        Initializes the TensorFlow session and all variables
        Args:
            None
        Returns:
            None
        '''
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.sess.run(self.global_variables_initializer)
        self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=self.graph)

    def _define_inputs(self):
        '''
        Initializes the placeholder Tensors used for constructing the computational graph, the model endpoints, and important parameters
        Args:
            None
        Returns:
            None
        '''
        if self.input_shape == None:
            self.input_shape = [None, 3, 224, 224] # use ImageNet defaults

        self.images = tf.placeholder(
            tf.float32,
            shape=self.input_shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

    def build(self, weight_dict):
        '''

        Takes in a dictionary containing the weights of the model and builds the computational graph of the TensorFlow model

        Args:
            weight_dict: a dictionary containing the weights for the model

        Returns:
            output: the output of the placeholder input Tensor of the model
        '''
        output = self.images

        for i, layer_attr in enumerate(self.net_config):
            # get the ith weight in the weight dictionary
            weight = weight_dict[list(weight_dict.keys())[i]]

            # pass the input through the next layer, adding it to the tf.Graph (hopefully)

            output = create_tf_layer_from_name(output, layer_attr, weight)

        return output




