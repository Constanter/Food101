import argparse
import cv2
import numpy as np
from sklearn import preprocessing
import torch
from torchvision import transforms, models

from utils import get_model, get_inference_transforms


def parse_args():
    """Create and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument('model', type=str,
                        help='Name of the model for inference')
    parser.add_argument('weight_path', type=str,
                        help='Path to the best model weight')
    parser.add_argument('img_path', type=str,
                        help='Path to the image')
    parser.add_argument('le_path', type=str,
                        help='Path to the label encoder array')
    parser.add_argument('--num-classes', type=int, default=101,
                        help='Number classes to predict')

    return parser.parse_args()


def inference(
        model: models,
        device: torch.device,
        img_path: str,
        transform: transforms,
        le_path: str) -> str:
    """
    Function for inference model on custom image

    Parameters
    ----------
    model: models
        Model that we want to test
    device: torch.device,
        device where will be inference
    img_path: str,
        Path to test image
    transform: transforms,
        Transforms need to correct work with image
    le_path: str
        Path to label encoder classes

    Returns
    -------
    str
        Predicted class
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = torch.from_numpy(np.expand_dims(img, 0)).to(device)
    le = preprocessing.LabelEncoder()
    le.classes_ = np.load(le_path)
    with torch.no_grad():
        y_preds = model(img)
        _, preds = torch.max(y_preds, 1)
        preds = preds.tolist()
    result = le.inverse_transform(preds)[0]

    return result


def main():
    """
    Entry point

    Returns
    -------

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    name = args.model
    weight_path = args.weight_path
    img_path = args.img_path
    num_classes = args.num_classes
    le_path = args.le_path
    transform = get_inference_transforms()
    model = get_model(name, num_classes, device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    result = inference(model, device, img_path, transform, le_path)
    print(f'Image from {img_path} has class - {result}')


if __name__ == '__main__':
    main()
