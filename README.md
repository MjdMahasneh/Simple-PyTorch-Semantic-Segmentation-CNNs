# Simple PyTorch implementation of Semantic Segmentation CNNs
PyTorch Implementation of Semantic Segmentation CNNs: This repository features key architectures (from scratch) like **UNet**, **DeepLabv3+**, **SegNet**, **FCN**, and **PSPNet**. It's crafted to provide a solid foundation for Semantic Segmentation tasks using PyTorch.

## Supported Architectures and Backbones

- **[UNet](https://arxiv.org/abs/1505.04597)**: No backbone needed.
- **[DeepLabv3+](https://arxiv.org/abs/1802.02611v3)**: Support for ResNet backbones (ResNet18, ResNet34, ResNet50, and ResNet101).
- **[PSPNet](https://arxiv.org/abs/1612.01105)**: Support for ResNet backbones (ResNet18, ResNet34, ResNet50, and ResNet101).
- **[FCN](https://arxiv.org/abs/1411.4038)**: Support for VGG backbones (VGG11, VGG13, VGG16, and VGG19).
- **[SegNet](https://arxiv.org/abs/1511.00561)**: No backbone needed.



<table>
  <tr>
    <td><img src="material/unet.png" alt="UNet Architecture" width="400"/><br>UNet Architecture</td>
    <td><img src="material/deeplabv3+.png" alt="DeepLabv3+ Architecture" width="400"/><br>DeepLabv3+ Architecture</td>
  </tr>
  <tr>
    <td><img src="material/segnet.png" alt="SegNet Architecture" width="400"/><br>SegNet Architecture</td>
    <td><img src="material/fcn.png" alt="FCN Architecture" width="400"/><br>FCN Architecture</td>
  </tr>
  <tr>
    <td colspan="2" align="center"><img src="material/pspnet.jpg" alt="PSPNet Architecture" width="800"/><br>PSPNet Architecture</td>
  </tr>
</table>






## Optimizers and Learning Rate Schedulers

- **Optimizers**: Adam, SGD, and RMSprop.
- **Learning Rate Schedulers**: StepLR, PolyLR, and ReduceLROnPlateau.

## Loss Functions

- **Cross Entropy (variations)**:
  -  Standard CE
  - CE with class weights
  - Focal Loss
- **Dice Loss**
- **Joint loss**: Conjunction of Dice loss with one of the CE variations (as a Joint Loss). 

## Evaluation Metrics

- Models are evaluated using:
  - **Dice** Coefficient
  - Intersection over Union **(IoU)** score.

## Dataset

A mock dataset is included in the repository for demonstration and testing purposes. Note that this dataset is not aimed to be used for training/testing, but rather for setting up and debugging for the first run, a convenience.

Replace the mock dataset with your own dataset as needed. The data loader accepts images of arbitrary dimensions and resizes them to the target size. Ensure that your dataset follows the below directory structure for optimal compatibility with the data loader:
```
root:
│
├── train
│ ├── images
│ └── masks
│
├── val
│ ├── images
│ └── masks
```

This structure includes separate subfolders for training and validation data, with further subdivisions for images and their corresponding masks.

When preparing your dataset, ensure that the images are in `.jpg` and masks are in `.png` format and are of the same size as their corresponding images. Each RGB pixel in the mask should represent a class label as an integer. For instance, in a dataset with 3 classes, use [0,0,0], [1,1,1], and [2,2,2] to label these classes. 

For a better understanding of the expected data structure and mask format, please refer to the mock dataset included in this repository. The mock dataset serves as a practical example, demonstrating how your data and masks should be organized and formatted, use `./mock_dataset/inspect_data.py` to visualize masks and images.

Edit `utils/data_loading.py` to modify behavior. 



## Requirements

To run this project, you need to have the following packages installed:

- `torch`
- `matplotlib`
- `numpy`
- `Pillow`
- `tqdm`
- `torchvision`
- `opencv-python`

You can install them by running the following command:

```bash
pip install -r requirements.txt
```
Alternatively, you can manually install each package using:
```bash
pip install torch matplotlib numpy Pillow tqdm torchvision
```


## Training:

Modify the `config.py` file as needed, including dataset paths:

```     self.epochs = 5  # Number of training epochs
        self.batch_size = 2  # Batch size for training
        self.lr = 1e-5  # Learning rate
        self.optimization = 'RMSprop'  # Optimization method ('RMSprop', 'SGD', 'Adam')
        self.lr_policy = 'plateau'  # Learning rate policy ('plateau', 'poly', 'step')
        self.lr_decay_step = 0.5  # LR decay step for 'step' policy

        self.load = False  # Flag to load model from a .pth file
        self.val_frequency = 20  # Validation frequency as a percentage
        self.extra_weight_frequency = 10  # Frequency for saving extra weights as a percentage

        self.amp = False  # Use mixed precision training
        self.classes = 3  # Number of output classes
        self.target_size = (512, 512)  # Target size for input images (height, width)
        self.number_of_in_channels = 3  # Number of input channels

        self.loss_type = 'joint'  # Loss type ('dice', 'ce', 'joint')
        self.CE_variation = 'ce'  # Cross-entropy variation ('CE', 'CEW', 'Focal')
        self.class_weights = None  # Class weights for CE loss or a list of weights for each class e.g., [1.0, 1.0, 4.0] for [class1, class2, class3]

        # Directories for dataset and checkpoints
        self.dir_root = Path('F:/projects/semantic_segmentaion_archs_repo/mock_dataset')  # Root directory for dataset
        self.train_images_dir = Path(os.path.join(self.dir_root, 'train/images'))
        self.train_mask_dir = Path(os.path.join(self.dir_root, 'train/masks'))
        self.val_images_dir = Path(os.path.join(self.dir_root, 'val/images'))
        self.val_mask_dir = Path(os.path.join(self.dir_root, 'val/masks'))
        self.dir_checkpoint = Path('./checkpoints/')  # Directory for saving checkpoints

        # CNN architecture and backbone
        self.cnn_arch = 'deeplabv3+'  # CNN architecture ('UNet', 'DeepLab', 'SegNet', 'PSPNet', 'FCN')
        self.backbone = 'resnet18'  # Backbone model ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'vgg11', 'vgg13', 'vgg16', 'vgg19')
        self.use_pretrained_backbone = True  # Use a pretrained backbone
        self.bilinear = True  # Use bilinear upsampling
```

then Run train.py directly or Execute the training script using the following command:

```bash
python train.py
```

Note compatibility when creating segmentation models:

| Architecture | Compatible Backbones |
|--------------|----------------------|
| FCN Variants |  |
| - FCN32s     | VGG11, VGG13, VGG16, VGG19 |
| - FCN16s     | VGG11, VGG13, VGG16, VGG19 |
| - FCN8s      | VGG11, VGG13, VGG16, VGG19 |
| - FCNs       | VGG11, VGG13, VGG16, VGG19 |
| DeepLabv3+   | ResNet18, ResNet34, ResNet50, ResNet101 |
| PSPNet       | ResNet18, ResNet34, ResNet50, ResNet101 |
| UNet         | None (No backbone required) |
| SegNet       | None (No backbone required) |


## Prediction:
Modify the attributes of the Args class in `predict.py` file as needed, (including path for testing images, model weights, and prediction network parameters), then Run predict.py directly or Execute the prediction script using the following command:

```bash
python predict.py
```


## Contributing

Contributions to this repository are welcome. Feel free to submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the terms of the MIT license.

