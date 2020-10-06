import torchvision.models as models

def get_model_arch(model_str):
    if model_str == 'resnet18':
        return models.resnet18(pretrained=True)
    #TODO add more models
    else:
        return NotImplementedError
