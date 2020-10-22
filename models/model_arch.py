import torchvision.models as models

def get_model_arch(model_str):
    '''
    Returns the appropriate PyTorch model for a given model name

    Args:
        model_str: a string describing the model to return
    
    Returns:
        the needed torchvision.models pretrained model
    '''
    if model_str == 'resnet18':
        return models.resnet18(pretrained=True)
    elif model_str == 'alexnet':
        return models.alexnet(pretrained=True)
    elif model_str == 'squeezenet':
        return models.squeezenet1_0(pretrained=True)
    elif model_str == 'vgg16':
        return models.vgg16(pretrained=True)
    elif model_str == 'densenet':
        return models.densenet161(pretrained=True)
    elif model_str == 'inception':
        return models.inception_v3(pretrained=True)
    elif model_str == 'googlenet':
        return models.googlenet(pretrained=True)
    elif model_str == 'shufflenet':
        return models.shufflenet_v2_x1_0(pretrained=True)
    elif model_str == 'mobilenet':
        return models.mobilenet_v2(pretrained=True)
    elif model_str == 'resnext50_32x4d':
        return models.resnext50_32x4d(pretrained=True)
    elif model_str == 'wide_resnet50_2':
        return models.wide_resnet50_2(pretrained=True)
    elif model_str == 'mnasnet':
        return models.mnasnet1_0(pretrained=True)
    else:
        return NotImplementedError
