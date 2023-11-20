# Simple-PyTorch-implementation-of-Semantic-Segmentation-CNNs
PyTorch Implementation of Semantic Segmentation CNNs: This repository features key architectures (from scratch) like UNet, DeepLabv3+, SegNet, FCN, and PSPNet. It's crafted to provide a solid foundation for Semantic Segmentation tasks using PyTorch.

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

- **Cross Entropy (variations)**: Standard CE, CE with class weights, and Focal Loss.
- **Dice Loss**: Standalone or in conjunction with one of the CE variations (as a Joint Loss).

## Evaluation Metrics

- Models are evaluated using either Dice Coefficient or Intersection over Union (IoU) score.

## Dataset

A mock dataset is included in the repository for demonstration and testing purposes. Note that this dataset is not aimed to be used for training/testing, but rather for setting up and debugging for the first run, a convenience.


## Requirements

To run this repository, you will need to install certain dependencies. If you are using a conda environment, you can find out your environment's requirements using the command `conda list`. This will list all the packages and their versions installed in your current conda environment.

## How to Run

1. **Install Requirements**: Install the necessary dependencies.
   ```bash
   pip install -r requirements.txt


## Training

Modify the `config.py` file as needed, including dataset paths, then Run train.py directly or Execute the training script using the following command:

```bash
python train.py
```

## Contributing

Contributions to this repository are welcome. Feel free to submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the terms of the MIT license.

