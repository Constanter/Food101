import cv2
from pathlib import Path
from sklearn import preprocessing
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any


class FoodData(Dataset):
    def __init__(self, img_path, img_dir, size, le, transform=None):
        self.img_path = img_path
        self.img_dir = img_dir.as_posix()
        self.transform = transform
        self.size = size
        self.__le = le

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        label, img_name = self.img_path[index].split('/')
        path = self.img_dir + '/images/' + label + '/' + img_name + '.jpg'
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return {
            'image': img,
            'label': torch.tensor(self.__le.transform([label])[0])
        }


class Dataloaders:
    """
    Class create from raw data DataLoader
    """
    def __init__(self, root: Path):
        """
        Initialize main paths

        Parameters
        ----------
        root: Path
            Path to the root folder with dataset
        """
        self.__root = root
        self.__images_path = root / 'images'
        self.__class_path = root / 'meta' / 'classes.txt'
        self.__train_img_pth = root / 'meta' / 'train.txt'
        self.__test_img_pth = root / 'meta' / 'test.txt'
        self.__lbl_enc = preprocessing.LabelEncoder()

    @staticmethod
    def __file2list(path: Path) -> list[str]:
        """
        Convert .txt file to the list, where each line represent path to the image.

        Parameters
        ----------
        path: Path
            Path to the .txt file

        Returns
        -------
        list[str]
            List where each element is string with relevant path the image
        """
        with open(path, 'r') as file1:
            lines = file1.readlines()
        final_list = [line.strip() for line in lines]
        return final_list

    @staticmethod
    def __get_augmentations(size: int, crop_size: int) -> tuple[transforms, transforms]:
        """
        Get augmentation for the train process

        Parameters
        ----------
        size: int
            Size that image will be resize
        crop_size: int
            Size of the central crop

        Returns
        -------
        tuple[transforms, transforms]
            Train, test transforms
        """
        transforms_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=size),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        transforms_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=size),
            transforms.CenterCrop(size=crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        return transforms_train, transforms_test

    def __call__(self, config: dict[str, Any]) -> tuple[dict, dict]:
        """
        Function convert raw data/

        Parameters
        ----------
        config: dict
            Configuration of dataloader
        Returns
        -------
        tuple[dict, dict]
            Return dict with datasets and datasets sizes
        """
        classes = self.__file2list(self.__class_path)
        train_data = self.__file2list(self.__train_img_pth)
        test_data = self.__file2list(self.__test_img_pth)
        self.__lbl_enc.fit_transform(classes)
        img_size = config['img_size']
        center_crop_size = config['center_crop_size']

        train_aug, test_aug = self.__get_augmentations(img_size, center_crop_size)

        train_dataset = FoodData(train_data, self.__root, img_size, self.__lbl_enc, train_aug)
        test_dataset = FoodData(test_data, self.__root, img_size, self.__lbl_enc, test_aug)

        batch = config['batch_size']

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

        dataloaders = dict()
        dataset_sizes = dict()

        dataloaders['train'] = train_loader
        dataloaders['val'] = test_loader
        dataset_sizes['train'] = train_dataset.__len__()
        dataset_sizes['val'] = test_dataset.__len__()

        return dataloaders, dataset_sizes
