import os
from pathlib import Path



class Config:

    """
    Configuration class for training.
    Usage:
        args = Config()
        print(vars(args))
        print(args.epochs)
    """

    def __init__(self):
        self.epochs = 5  # Number of training epochs
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

        """
        Available Architectures and Compatible Backbones:

        1. FCN Variants (Fully Convolutional Networks):
           - FCN32s: Compatible with VGG11, VGG13, VGG16, VGG19
           - FCN16s: Compatible with VGG11, VGG13, VGG16, VGG19
           - FCN8s: Compatible with VGG11, VGG13, VGG16, VGG19
           - FCNs: Compatible with VGG11, VGG13, VGG16, VGG19

        2. DeepLabv3+:
           - Compatible with ResNet18, ResNet34, ResNet50, ResNet101

        3. PSPNet (Pyramid Scene Parsing Network):
           - Compatible with ResNet18, ResNet34, ResNet50, ResNet101

        4. UNet:
           - Takes None. Does not require a backbone.

        5. SegNet:
           - Takes None. Does not require a backbone.
        """

        # CNN architecture and backbone
        self.cnn_arch = 'deeplabv3+'  # CNN architecture ('UNet', 'DeepLab', 'SegNet', 'PSPNet', 'FCN')
        self.backbone = 'resnet18'  # Backbone model ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'vgg11', 'vgg13', 'vgg16', 'vgg19')
        self.use_pretrained_backbone = True  # Use a pretrained backbone
        self.bilinear = True  # Use bilinear upsampling




if __name__ == '__main__':

    from utils.validate_configuration import validate_config

    config = Config()

    validate_config(config)
    print("Configuration is valid.")
