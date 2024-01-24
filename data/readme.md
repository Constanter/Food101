# Food 101 dataset

## Dataset info
Dataset has next structure :

```
{food 101}/
├── train/
│   ├── {class1}/
│   ├── {class2}/
│   ├── ...
└── val/
    ├── {class1}/
    ├── {class2}/
    ├── ...
```

Train dataset consists of 75750 images divided in 101 class
Test dataset consists of 25250 images divided in 101 class

Each class in train dataset has 750 images:

![image](./eda_graphs/train_classes.png)

Each class in test dataset has 250 images:

![image](./eda_graphs/test_classes.png)

### Width distribution

Width distribution train dataset:

![image](./eda_graphs/train width.png)

Width distribution test dataset:

![image](./eda_graphs/test width.png)

### Height distribution

Height distribution train dataset:

![image](./eda_graphs/train height.png)

Height distribution test dataset:

![image](./eda_graphs/test height.png)

