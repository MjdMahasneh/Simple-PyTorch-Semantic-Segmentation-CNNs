
class ConfigError(Exception):
    """Raised when an incompatible configuration is chosen."""
    pass

def validate_config(config):
    """
    Validates the selected architecture, backbone, optimization method, learning rate policy, and loss type in the Config class.
    Raises an error if an incompatible combination is chosen.

    # Example usage
    config = Config()
    try:
        validate_config(config)
        print("Configuration is valid.")
    except ConfigError as e:
        print(f"Configuration Error: {e}")
    """
    # Define compatible options
    compatible_backbones = {
        'fcn32s': ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
        'fcn16s': ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
        'fcn8s': ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
        'fcns': ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
        'deeplabv3+': ['resnet18', 'resnet34', 'resnet50', 'resnet101'],
        'pspnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101'],
        'unet': None,  # Indicates no backbone required
        'segnet': None  # Indicates no backbone required
    }

    compatible_optimizations = ['rmsprop', 'sgd', 'adam']
    compatible_lr_policies = ['plateau', 'poly', 'step']
    compatible_losses = ['dice', 'ce', 'joint']
    compatible_ce_variations = ['ce', 'cew', 'focal']

    # Check architecture and backbone
    selected_arch = config.cnn_arch
    selected_backbone = config.backbone

    if selected_arch not in compatible_backbones:
        raise ConfigError(f"Unknown architecture: {selected_arch}")

    if compatible_backbones[selected_arch] is None:
        if selected_backbone is not None:
            raise ConfigError(f"The architecture {selected_arch} does not require a backbone. Use 'backbone = None' instead.")
        if config.use_pretrained_backbone is not None:
            raise ConfigError(f"The architecture {selected_arch} does not require a backbone. Use 'use_pretrained_backbone = None' instead.")


    elif selected_backbone not in compatible_backbones[selected_arch]:
        raise ConfigError(f"Incompatible backbone for {selected_arch}. Choose from {compatible_backbones[selected_arch]}")

    # Check optimization method
    if config.optimization.lower() not in compatible_optimizations:
        raise ConfigError(f"Incompatible optimization method. Choose from {compatible_optimizations}")

    # Check learning rate policy
    if config.lr_policy.lower() not in compatible_lr_policies:
        raise ConfigError(f"Incompatible learning rate policy. Choose from {compatible_lr_policies}")

    # Check loss type
    if config.loss_type.lower() not in compatible_losses:
        raise ConfigError(f"Incompatible loss type. Choose from {compatible_losses}")

    # Check CE variation
    if config.loss_type.lower() in ['ce', 'joint']:
        if config.CE_variation.lower() not in compatible_ce_variations:
            raise ConfigError(f"Incompatible CE variation for '{config.loss_type}' loss type. Choose from {compatible_ce_variations}")
        if config.CE_variation.lower() == 'focal' and config.classes <= 1:
            raise ConfigError("Focal loss is only implemented for multi-class classification (classes > 1).")
    else:
        if config.CE_variation is not None:
            raise ConfigError("CE_variation should be None when loss_type is not 'ce' or 'joint'.")

    # Check class weights
    if config.CE_variation is not None:
        if config.CE_variation.lower() == 'cew':
            if config.class_weights is None:
                raise ConfigError("Class weights must be provided for 'CEW' variation.")
            elif len(config.class_weights) != config.classes:
                raise ConfigError("Length of class weights must match the number of classes.")
        else:
            if config.class_weights is not None:
                raise ConfigError("Class weights should be None unless 'CEW' variation is chosen.")






if __name__ == '__main__':
    class Config:
        '''Configuration class for training
            Usage:
                args = Config()
                print(vars(args))
                print(args.epochs)
        '''
        def __init__(self):
            # Number of epochs for training. Default is 5.
            self.epochs = 5

            # Size of each input batch. Default is 1.
            self.batch_size = 2

            # Learning rate for optimization. Default is 1e-5.
            self.lr = 1e-5

            self.optimization = 'rmsprop' # Optimization method. choose from 'RMSprop', 'SGD' or 'Adam', Default is 'RMSprop'

            self.lr_policy = 'plateau' # Learning rate policy. choose from 'plateau', 'poly' or 'step', Default is 'plateau'.

            self.lr_decay_step = 0.5 # Decay step for the learning rate (e.g., will decay after half total_iterations when set to 0.5). Default is 0.5. Only used when lr_policy is 'step'

            # Whether to load the model from a pre-existing .pth file. Default is False.
            self.load = False


            ## (DONE) to-do: proceess deprecated arguments scale/val
            # Downscaling factor for the input images. Default is 0.5.
            # self.scale = 0.5 ## -> deprecated: replaced with target_size

            # Percentage of data to be used for validation. Values range between 0-100. Default is 10.0.
            self.val = 10.0 ## -> deprecated: replaced with

            # Whether to use mixed precision during training. Default is False.
            self.amp = False

            # Whether to use bilinear upsampling. Default is False.
            self.bilinear = False

            # Number of output classes. Default is 2.
            self.classes = 3 #2

            self.target_size = (512, 512) ## (height, width)

            self.loss_type = 'joint' ## 'dice' or 'ce' or 'joint'

            self.CE_variation = 'focal' ## 'CE', 'CEW' (CE with class_weights), or 'Focal'. Note Focal is only implemented for multi-class classification for now, so can only be used if self.classes > 1.

            self.class_weights = None #[1.0, 1.0, 4.0] #[BG, pot, plant] ## either None or a list of weights for each class. Note: only useful if cls_criterion_variation is set to = 'CE', otherwise ignored.

            self.extra_weight_interval = 150 ## 150 ## None or integer between 1 and number of batches in the dataset (e.g., if an epoch has 100 batches, then the interval can be between 1 and 100). Usefull for saving checkpoints at different intervals than the end of each epoch

            ## initialize the paths
            # self.dir_root = Path('G:/Datasets/carvana-image-masking-challenge')
            # self.dir_root = Path('G:/Datasets/MoA_STC_dataset_export-20230630_170319 - II/MoA_semantic_seg_subset-1.0')
            # # self.dir_img = Path(os.path.join(self.dir_root, 'images'))
            # # self.dir_mask = Path(os.path.join(self.dir_root, 'masks'))
            # self.train_images_dir = Path(os.path.join(self.dir_root, 'train/images'))
            # self.train_mask_dir = Path(os.path.join(self.dir_root, 'train/masks'))
            # self.val_images_dir = Path(os.path.join(self.dir_root, 'val/images'))
            # self.val_mask_dir = Path(os.path.join(self.dir_root, 'val/masks'))

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
            self.cnn_arch = 'fcn8s' #'DeepLab' ## 'UNet' or 'DeepLab' or 'SegNet' or 'PSPNet' or 'FCN'

            # self.dir_checkpoint = Path('./checkpoints - temp - weighted CE + extra weights/')
            #self.dir_checkpoint = Path('./checkpoints - temp - DeepLab/testtttinngggg/')

            ## vgg11, vgg13, vgg16, vgg19
            self.backbone = 'vgg16' #None #'resnet18' ## 'resnet18' 'resnet34' or 'resnet50' or 'resnet101'


    config = Config()
    try:
        validate_config(config)
        print("Configuration is valid.")
    except ConfigError as e:
        print(f"Configuration Error: {e}")