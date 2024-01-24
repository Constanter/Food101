import copy
import time
import torch
from torch import nn, device
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from typing import Any


def train_model(
        model: models,
        criterion: nn,
        optimizer: optim,
        dataloaders: dict[str, DataLoader],
        dataset_sizes: dict[str, int],
        scheduler: lr_scheduler,
        dev: device,
        num_epochs: int = 10
) -> dict[str, Any]:
    """
    Functional for training model

    Parameters
    ----------
    model: results,
        Model for training
    criterion: nn,
        Metric function
    optimizer: optim,
        Optimizer function
    dataloaders: dict[str, DataLoader],
        Dataloaders with data for train, test
    dataset_sizes: dict[str, int],
        Sizes of datasets
    scheduler: lr_scheduler,
        Scheduler function
    dev: device,
        Device ['cpu', 'gpu']
    num_epochs: int,
        Number of epochs

    Returns
    -------
    dict
        Dictionary with results of the training
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train': {'loss': [], 'acc': [], 'epoch': []}, 'val': {'loss': [], 'acc': [], 'epoch': []}}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for data in tqdm(dataloaders[phase]):
                inputs = data['image'].squeeze(0).to(dev)
                labels = data['label'].to(dev)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
                optimizer.param_groups[0]['lr'] /= 2

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(float(epoch_acc.cpu()))
            history[phase]['epoch'].append(epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    history['model'] = best_model_wts

    return history


def get_train_properties(config: dict[str, Any], model: models) -> tuple[optim, lr_scheduler, nn]:
    """
    Create function for training

    Parameters
    ----------
    config: dict
        Config with parameters of training
    model: models
        Model that will be trained

    Returns
    -------
    Tuple with functions for training
    """
    train_mode = config['mode']
    optim_func = config['optimizer']
    criterion = nn.CrossEntropyLoss()
    lr = config['lr']
    if train_mode == 'transfer':
        for params in list(model.parameters())[0:-5]:
            params.requires_grad = False

    if optim_func == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optim_func == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optim_func == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise NotImplementedError

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    return optimizer, exp_lr_scheduler, criterion
