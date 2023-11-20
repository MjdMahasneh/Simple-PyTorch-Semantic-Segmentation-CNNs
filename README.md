# Simple-PyTorch-implementation-of-Semantic-Segmentation-CNNs
PyTorch Implementation of Semantic Segmentation CNNs: This repository features key architectures (from scratch) like UNet, DeepLabv3+, SegNet, FCN, and PSPNet. It's crafted to provide a solid foundation for Semantic Segmentation tasks using PyTorch.

## Supported Architectures and Backbones

- **DeepLabv3+**: Support for ResNet backbones (ResNet18, ResNet34, ResNet50, and ResNet101).
- **PSPNet**: Support for ResNet backbones (ResNet18, ResNet34, ResNet50, and ResNet101).
- **FCN**: Support for VGG backbones (VGG11, VGG13, VGG16, and VGG19).
- **UNet**: No backbone needed.
- **SegNet**: No backbone needed.



<img src="materials/unet.png" alt="UNet Architecture" width="200"/>
<img src="materials/deeplabv3+.png" alt="DeepLabv3+ Architecture" width="200"/>
<img src="materials/segnet.png" alt="SegNet Architecture" width="200"/>
<img src="materials/fcn.png" alt="FCN Architecture" width="200"/>
<img src="materials/pspnet.png" alt="PSPNet Architecture" width="200"/>
![UNet Architecture](materials/unet.png)
![DeepLabv3+ Architecture](materials/deeplabv3plus_architecture.png)
![SegNet Architecture](materials/segnet_architecture.png)
![FCN Architecture](materials/fcn_architecture.png)
![PSPNet Architecture](materials/pspnet_architecture.png)

## References

- [UNet](https://arxiv.org/abs/1505.04597)
- [FCN](https://arxiv.org/abs/1411.4038)
- [DeepLabv3+](https://arxiv.org/abs/1802.02611v3)
- [PSPNet](https://arxiv.org/abs/1612.01105)
- [SegNet](https://arxiv.org/abs/1511.00561)

## Optimizers and Learning Rate Schedulers

- **Optimizers**: Adam, SGD, and RMSprop.
- **Learning Rate Schedulers**: StepLR, PolyLR, and ReduceLROnPlateau.

## Loss Functions

- **Cross Entropy**: Standard CE, CE with class weights, and Focal Loss.
- **Dice Loss**: Standalone or in conjunction with one of the CE variations (Joint Loss).

## Evaluation Metrics

- Models are evaluated using either Dice Coefficient or Intersection over Union (IoU) score.

## Dataset

A mock dataset is included in the repository for demonstration and testing purposes. Note that this dataset is not aimed to be used for training/testing, but rather for setting up and debugging for the first run, a convenience.

## Network Flowcharts and References

- Flowcharts depicting the architecture of each network are provided.
- References to the original papers for each architecture are included.

![Architecture Flowchart](temp_name.png)

## Requirements

To run this repository, you will need to install certain dependencies. If you are using a conda environment, you can find out your environment's requirements using the command `conda list`. This will list all the packages and their versions installed in your current conda environment.

## How to Run

1. **Install Requirements**: Install the necessary dependencies.
   ```bash
   pip install -r requirements.txt

## Configuration

Modify the `config.py` file as needed, including dataset paths.

## Run Training

Execute the training script using the following command:

```bash
python train.py
```




## Contributing

Contributions to this repository are welcome. Feel free to submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the terms of the MIT license.

