import cv2
from collections import Counter
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_images_paths(path: Path) -> list[str]:
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


def get_stats(data: list[str], root: Path) -> tuple[list, list, Counter]:
    """
    Calculate statistics for height, width, class distribution
    Parameters
    ----------
    data
        Relevant paths to the images
    root
        Absolute path to the dataset folder

    Returns
    -------
    tuple with statistics
    """
    sizes, labels = [], []
    for img_path in tqdm(data):
        label, img_name = img_path.split('/')
        img_name += '.jpg'
        path = root / 'images' / label / img_name
        img = cv2.imread(path.as_posix())
        sizes.append(img.shape[:2])
        labels.append(label)
    height = [size[0] for size in sizes]
    width = [size[1] for size in sizes]
    cnt = Counter(labels)
    return height, width, cnt


def get_image_size_distribution(root: Path) -> None:
    """
    Create plots from image statistics

    Parameters
    ----------
    root
        Absolute path to the dataset folder
    Returns
    -------

    """
    train_img_pth = root / 'meta' / 'train.txt'
    test_img_pth = root / 'meta' / 'test.txt'
    train_data = get_images_paths(train_img_pth)
    test_data = get_images_paths(test_img_pth)

    train_height, train_width, label_cnt_train = get_stats(train_data, root)
    test_height, test_width, label_cnt_test = get_stats(test_data, root)
    plot_distribution(test_height, 'test height')
    plot_distribution(test_width, 'test width')
    plot_distribution(train_height, 'train height')
    plot_distribution(train_width, 'train width')
    plot_class_distribution(label_cnt_train)
    plot_class_distribution(label_cnt_test, False)


def plot_class_distribution(data: dict[str: int], train: bool = True) -> None:
    """
    Plot classes distribution

    Parameters
    ----------
    data: dict
        Dictionary where key class name, value number class elements
    train: bool
        True if from train dataset

    Returns
    -------
    None
    """
    name = 'train_classes' if train else 'test_classes'

    plt.bar(list(data.keys()), data.values(), color='g')
    plt.xticks(rotation=90)
    plt.savefig(f'./eda_graphs/{name}.png')
    plt.close()


def plot_distribution(data: list[str], name: str) -> None:
    """
    Plot distribution of variable

    Parameters
    ----------
    data: list
        List with variable distribution
    name: str
        Name of parameter

    Returns
    -------
    None
    """
    bins = range(0, 1000, 50)
    plt.xlim([0, 1000])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title(f'Distribution of {name}')
    plt.xlabel('Values')
    plt.ylabel('count')

    plt.savefig(f'./eda_graphs/{name}.png')
    plt.close()


if __name__ == "__main__":
    get_image_size_distribution(Path('~/data/food/dataset/food-101/'))
