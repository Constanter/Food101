import numpy as np
import cv2
from torchvision import models, transforms
import torch
from torch import nn
from sklearn import preprocessing
import importlib.resources

NUM_CLS = 101
IMG_SIZE = 256
CROP_SIZE = 224


def inference(img_path: str) -> None:
    """
    Function for predicting image class

    Parameters
    ----------
    img_path
        Input image path
    Returns
    -------
    None
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
    weight_path = 'model.pth'
    le_path = 'classes_food.npy'
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
    original_img = cv2.imread(img_path)
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    thickness = 2
    original_img = cv2.resize(original_img, (512, 512))
    image = cv2.putText(original_img, result, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.imwrite(img_path, image)
