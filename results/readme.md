## Food101 - Classification

Model compare 

### Training results

|                     | Accuracy % | Size     | Training Mode |
|---------------------|------------|----------|-----|
| **Resnet18**        | 57.36      | 44.8 MB  |  finetune |
| **Efficientnet b5** | 48.61      | 173.4 MB |  finetune |
| **Swin-s**          | 78.3       | 197.7 MB | finetune |

**Batch size**: 64, 

All models were trained for **10 epochs**;


### Training graphs

**Resnet18:-** 

Finetuning the pretrained resnet18 model.
![Screenshot](resnet18/acc.png)
![Screenshot](resnet18/loss.png)

**Efficientnet_b5:-** 

Finetuning the pretrained efficientnet_b6 model.
![Screenshot](efficientnet_b6/acc.png)
![Screenshot](efficientnet_b6/loss.png)

**Swin-s:-** 

Finetuning the pretrained swin-s model.
![Screenshot](swin-s/acc.png)
![Screenshot](swin-s/loss.png)



### Todo

1. Experiments with different **learning-rate and optimizers**.
2. Train models from scratch
3. Use label smoothing loss
