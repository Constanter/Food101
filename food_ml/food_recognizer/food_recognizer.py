import importlib.resources
import cv2
import numpy as np
from sklearn import preprocessing
import torch
from torch import nn
from torchvision import models, transforms


with importlib.resources.path('food_recognizer.data', 'model.pth') as p:
    weight_path = p.as_posix()

with importlib.resources.path('food_recognizer.data', 'classes_food.npy') as p:
    le_path = p.as_posix()

NUM_CLS = 101
IMG_SIZE = 256
CROP_SIZE = 224


def inference(img: np.ndarray) -> str:
    """
    Function for predicting image class

    Parameters
    ----------
    img
        Input image from source
    Returns
    -------
    Class of image
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.swin_s()
    n_inputs = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, NUM_CLS)
    )

    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=IMG_SIZE),
        transforms.CenterCrop(size=CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    original_img = img
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
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
