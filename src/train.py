import argparse
from pathlib import Path
import torch

from utils import (
    Dataloaders, train_model, get_train_properties,
    get_model, get_config, seed_everything, save_graphs
)


def parse_args():
    """Create and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument('--config', type=Path, default='./config.yaml',
                        help='Path to the config file')

    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args.config)
    model_config = config['model']
    train_config = config['train']
    seed_everything(seed=train_config['seed'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(**model_config, dev=device)
    root_path = Path(train_config['root_path'])
    loader = Dataloaders(root_path)
    dataloaders, dataset_sizes = loader(train_config)
    optimizer, scheduler, criterion = get_train_properties(train_config, model)

    results = train_model(
        model, criterion, optimizer, dataloaders,
        dataset_sizes, scheduler, device
    )
    save_path = Path.cwd().parent / 'results' / model_config['name']
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / 'best_model.pth'
    torch.save(results['model'], model_path)
    save_graphs(results, save_path)


if __name__ == "__main__":
    main()
