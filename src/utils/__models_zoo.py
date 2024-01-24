from torch import nn, device
from torchvision import models


def get_model(name: str, num_classes: int, dev: device) -> models:
    """
    Function create model by name from config

    Parameters
    ----------
    name: str
        Name of the model
    num_classes: 101
        Number of classes for predicted
    dev: device
        Device for inference ['cpu', 'cuda']
    Returns
    -------
    models
        Model with parameters from config
    """
    if 'resnet' in name:
        if name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif name == 'resnet152':
            model = models.resnet152(pretrained=True)
        else:
            raise NotImplementedError
        # freeze all model parameters
        for param in model.parameters():
            param.requires_grad = False

        # new final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif 'efficientnet' in name:
        if name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        elif name == 'efficientnet_b6':
            model = models.efficientnet_b6(weights='IMAGENET1K_V1')
        else:
            raise NotImplementedError
        for params in model.parameters():
            params.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)

    elif 'swin' in name:
        if name == 'swin-s':
            model = models.swin_s(pretrained=True)
        elif name == 'swin-b':
            model = models.swin_b(pretrained=True)
        else:
            raise NotImplementedError
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    else:
        raise NotImplementedError

    model.to(dev)
    return model
