from pathlib import Path
import yaml
import os
import random
from matplotlib.pylab import plt
import numpy as np
import torch
from torchvision import transforms


def get_config(pth: Path) -> dict:
    """
    Open yaml config and generate dictionary

    Parameters
    ----------
    pth: Path
        Path to the config

    Returns
    -------
    dict
        Config where key - parameter, value - it's value
    """
    with Path.open(pth, 'r') as fd:
        config = yaml.safe_load(fd)
    return config


def seed_everything(seed: int = 42) -> None:
    """
    Make seed for experiments reproducibility

    Parameters
    ----------
    seed: int
        Seed number

    Returns
    -------
    None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_inference_transforms(img_size=256, crop_size=224) -> transforms:
    """
    Create transforms for correct model inference

    Parameters
    ----------
    img_size: int
        Size that image will be resize
    crop_size: int
        Size of the central crop

    Returns
    -------
    Inference transforms
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=img_size),
        transforms.CenterCrop(size=crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform


def save_graphs(history: dict[str, dict], save_path: Path) -> None:
    """
    Save training graphs

    Parameters
    ----------
    history: dict
        Result of training
    save_path: Path
        Path where will be saved results

    Returns
    -------
    None
    """
    for metr in ['loss', 'acc']:
        train_values = history['train'][metr]
        val_values = history['val'][metr]

        epochs = history['val']['epoch']

        # Plot and label the training and validation loss values
        plt.plot(epochs, train_values, label=f'Training {metr}')
        plt.plot(epochs, val_values, label=f'Validation {metr}')

        # Add in a title and axes labels
        plt.title(f'Training and Validation {metr}')
        plt.xlabel('Epochs')
        plt.ylabel(f'{metr}')

        plt.legend(loc='best')
        name = save_path / f'{metr}.png'
        plt.savefig(name)
        plt.close()
