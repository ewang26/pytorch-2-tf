import torch
import torch.nn as nn

def module_to_dict(module):
    """
    Converts PyTorch module to dictionary of parameters.

    Args:
        module (nn.Module): The module to be converted

    Returns:
        Dict[str, object]: Dictionary of module's parameters, where the name of the parameter is the `str` and the `object` is the parameter's value. Returns None if module is unsupported.
    """

    # Convolutions
    if isinstance(module, nn.Conv1d):
        return {
            'name': module.__class__.__name__,
            'in_channels': module.in_channels,
            'out_channels': module.out_channels,
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
        }
    elif isinstance(module, nn.Conv2d):
        return {
            'name': module.__class__.__name__,
            'in_channels': module.in_channels,
            'out_channels': module.out_channels,
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
        }

    # Pooling
    elif isinstance(module, nn.MaxPool1d):
        return {
            'name': module.__class__.__name__,
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
            'dilation': module.dilation,
        }
    elif isinstance(module, nn.MaxPool2d):
        return {
            'name': module.__class__.__name__,
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
            'dilation': module.dilation,
        }
    elif isinstance(module, nn.AvgPool1d):
        return {
            'name': module.__class__.__name__,
            'kernel_size': kernel_size,
            'stride': module.stride,
            'padding': module.padding,
        }
    elif isinstance(module, nn.AvgPool2d):
        return {
            'name': module.__class__.__name__,
            'kernel_size': kernel_size,
            'stride': module.stride,
            'padding': module.padding,
        }

    # Normalization
    elif isinstance(module, nn.BatchNorm1d):
        return {
            'name': module.__class__.__name__,
            'num_features': module.num_features,
        }
    elif isinstance(module, nn.BatchNorm2d):
        return {
            'name': module.__class__.__name__,
            'num_features': module.num_features,
        }
    elif isinstance(module, nn.LayerNorm):
        return {
            'name': module.__class__.__name__,
            'normalized_shape': module.normalized_shape,
        }

    # Activation layers
    elif isinstance(module, nn.ReLU):
        return {
            'name': module.__class_.__name__,
        }
    elif isinstance(module, nn.LeakyReLU):
        return {
            'name': module.__class_.__name__,
            'negative_slope': negative_slope,
        }
    elif isinstance(module, nn.Sigmoid):
        return {
            'name': module.__class_.__name__,
        }
    elif isinstance(module, nn.Tanh):
        return {
            'name': module.__class_.__name__,
        }
    elif isinstance(module, nn.Softmax):
        return {
            'name': module.__class_.__name__,
            'dim': module.dim
        }
    elif isinstance(module, nn.LogSoftmax):
        return {
            'name': module.__class_.__name__,
            'dim': module.dim
        }

    # Linear
    elif isinstance(module, nn.Linear):
        return {
            'name': module.__class__.__name__,
            'in_features': module.in_features,
            'out_features': module.out_features,
            'bias': module.bias,
        }
    elif isinstance(module, nn.Identity):
        return {
            'name': module.__class__.__name__,
        }

    # Dropout
    elif isinstance(module, nn.Dropout):
        return {
            'name': module.__class__.__name__,
            'p': module.p,
        }

    # Loss
    elif isinstance(module, nn.CrossEntropyLoss):
        return {
            'name': module.__class__.__name__,
        }

    else:
        return None

