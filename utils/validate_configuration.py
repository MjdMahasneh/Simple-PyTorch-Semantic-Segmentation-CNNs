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
            raise ConfigError(
                f"The architecture {selected_arch} does not require a backbone. Use 'backbone = None' instead.")
        if config.use_pretrained_backbone is not None:
            raise ConfigError(
                f"The architecture {selected_arch} does not require a backbone. Use 'use_pretrained_backbone = None' instead.")


    elif selected_backbone not in compatible_backbones[selected_arch]:
        raise ConfigError(
            f"Incompatible backbone for {selected_arch}. Choose from {compatible_backbones[selected_arch]}")

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
            raise ConfigError(
                f"Incompatible CE variation for '{config.loss_type}' loss type. Choose from {compatible_ce_variations}")
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
            self.epochs = 5

            self.batch_size = 2

            self.lr = 1e-5

            self.optimization = 'rmsprop'

            self.lr_policy = 'plateau'

            self.lr_decay_step = 0.5

            self.load = False

            self.val = 10.0

            self.amp = False

            self.bilinear = False

            self.classes = 3

            self.target_size = (512, 512)

            self.loss_type = 'joint'

            self.CE_variation = 'focal'

            self.class_weights = None

            self.extra_weight_interval = 150

            self.cnn_arch = 'fcn8s'

            self.backbone = 'vgg16'


    config = Config()
    try:
        validate_config(config)
        print("Configuration is valid.")
    except ConfigError as e:
        print(f"Configuration Error: {e}")
