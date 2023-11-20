import argparse
import logging
import os
from datetime import datetime
from shutil import copyfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from evaluate import evaluate_dice
from evaluate import evaluate_iou
from unet import UNet
from deeplab import DeepLabv3Plus
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
import matplotlib.pyplot as plt
from utils.utils import plot_training_progress, write_to_file
from utils.scheduler import PolyLR
from utils.focal_loss import FocalLoss
import config as cfg
from utils.validate_configuration import validate_config, ConfigError
from FCN.FCN import get_fcn_model as FCN
from PSPNet.PSPNet import PSPNet
from SegNet.SegNet import SegNet


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,

        val_frequency: float = 0.5,
        save_checkpoint: bool = True,

        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,

        train_images_dir=None,
        train_mask_dir=None,
        val_images_dir=None,
        val_mask_dir=None,
        dir_checkpoint=None,
        target_size: tuple = (None, None),
        loss_type='joint',
        class_weights=None,
        extra_weight_frequency=None,
        cnn_arch=None,
        backbone=None,
        optimization=None,
        lr_policy=None,
        lr_decay_step=0.5,
        CE_variation=None,
        log_file=None,
):
    assert train_images_dir is not None and train_mask_dir is not None, 'Please provide the path to the images directory'
    assert val_images_dir is not None and val_mask_dir is not None, 'Please provide the path to the images directory'
    assert loss_type in ['dice', 'ce', 'joint'], 'Please provide a valid loss type: dice, ce or joint'
    assert val_frequency >= 0.0 and val_frequency <= 1.0, 'Please provide a valid validation frequency: between 0.0 and 1.0'
    assert extra_weight_frequency >= 0.0 and extra_weight_frequency <= 1.0, 'Please provide a valid extra weight frequency: between 0.0 and 1.0'

    if class_weights is not None:
        assert len(class_weights) == model.n_classes, 'Please provide a list of weights, for each class'

    train_losses, iter_overall_iou_scores, iter_ids = [], [], []
    iter_per_class_iou_scores = {i: [] for i in range(model.n_classes)}

    train_dataset = BasicDataset(images_dir=train_images_dir, mask_dir=train_mask_dir, mask_suffix='',
                                 target_size=target_size, stage='train')
    val_dataset = BasicDataset(images_dir=val_images_dir, mask_dir=val_mask_dir, mask_suffix='',
                               target_size=target_size, stage='val')

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
        Loss type:       {loss_type}
        Class weights:   {class_weights}
        Extra weight interval: {extra_weight_frequency}
        CNN architecture: {cnn_arch}
        Backbone:        {backbone}
    ''')

    if optimization == 'RMSprop':

        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    elif optimization == 'SGD':

        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimization == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError('Please provide a valid optimization method: RMSprop, SGD or Adam')

    total_batches_per_epoch = len(train_loader)  # Number of batches per epoch
    total_iterations = epochs * total_batches_per_epoch
    if lr_policy == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    elif lr_policy == 'poly':
        ## For PolyLR: The max_iters should typically be the total number of training iterations, which is usually the number of epochs multiplied by the number of batches per epoch. If you don't have this value upfront, you can calculate it dynamically during training or set a default value that you know will be safe (like the number of epochs).
        scheduler = PolyLR(optimizer, max_iters=total_iterations, power=0.9)
    elif lr_policy == 'step':
        ## For StepLR: The step_size is usually set to a value where you want the learning rate to decay, such as after a certain number of epochs.
        # lr_decay_step = 0.5
        step_size_value = int(
            total_iterations * lr_decay_step)  # Example: decay the LR every half of the total iterations
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_value, gamma=0.1)
    else:
        raise NotImplementedError('Please provide a valid learning rate policy: plateau, poly or step.')
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if CE_variation is not None:
        if CE_variation.lower() == 'ce':
            criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        elif CE_variation.lower() == 'cew':
            assert class_weights is not None
            weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        elif CE_variation.lower() == 'focal':
            assert model.n_classes > 1, 'Focal loss is only implemented for multi-class classification for now and can only be used when self.classes > 1.'

            criterion = FocalLoss(ignore_index=None, size_average=True)
        else:
            raise NotImplementedError('Please provide a valid CE variation: CE, CEW or Focal')
    else:
        assert loss_type in ['dice'], 'Please provide a valid loss type.'

    global_step = 0
    # Determine the number of iterations after which validation should occur
    validation_interval = int(
        total_iterations * val_frequency / epochs) if val_frequency > 0 else 0  ## if val_frequency is 0, then validation will be disabled

    extra_weight_interval = int(
        total_iterations * extra_weight_frequency / epochs) if extra_weight_frequency > 0 else 0  ## if val_frequency is 0, then validation will be disabled

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        ep_iter_count = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                ep_iter_count += 1

                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    if loss_type == 'joint':
                        if model.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                    elif loss_type == 'dice':
                        if model.n_classes == 1:
                            loss = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            loss = dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                    elif loss_type == 'ce':
                        if model.n_classes == 1:

                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        else:
                            loss = criterion(masks_pred, true_masks)
                    else:
                        raise NotImplementedError('Please provide a valid loss type: dice, ce or joint')

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                train_losses.append(loss.item())

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if log_file is not None:
                    msg = f'Epoch: {epoch}, ep iter: {ep_iter_count}, global iter: {global_step}, images used so far: {ep_iter_count * batch_size} out of {n_train}, loss: {loss.item()} \n'
                    write_to_file(log_file, msg)

                if validation_interval > 0 and global_step % validation_interval == 0:

                    val_score_dice = evaluate_dice(model, val_loader, device, amp)
                    val_score_iou, classwise_scores_iou = evaluate_iou(model, val_loader, device, amp)

                    # Scheduler step
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_score_dice)
                    else:
                        scheduler.step()

                    logging.info('Validation Dice score :  {}'.format(val_score_dice))
                    logging.info('Validation IoU score  :  {}'.format(val_score_iou))

                    for i, cls_iou in enumerate(classwise_scores_iou):
                        logging.info(f'Class {i} IoU score: {cls_iou}')
                        iter_per_class_iou_scores[i].append(cls_iou)  ## collect per class IoU scores

                    iter_overall_iou_scores.append(val_score_iou)  ## collect overall IoU scores
                    iter_ids.append(global_step)  ## collect iteration ids

                if extra_weight_interval > 0 and global_step % extra_weight_interval == 0:

                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    state_dict = model.state_dict()

                    state_dict['mask_values'] = train_dataset.mask_values

                    weights_path = os.path.join(dir_checkpoint, 'checkpoint_epoch{}_{}.pth'.format(epoch, global_step))

                    if log_file is not None:
                        msg = f'saving weights : {weights_path} \n'
                        write_to_file(log_file, msg)

                    torch.save(state_dict, weights_path)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()

            state_dict['mask_values'] = train_dataset.mask_values

            weights_path = os.path.join(dir_checkpoint, 'checkpoint_epoch{}.pth'.format(epoch))
            if log_file is not None:
                msg = f'saving weights : {weights_path} \n'
                write_to_file(log_file, msg)
            torch.save(state_dict, weights_path)
            logging.info(f'Checkpoint {epoch} saved!')

        loss_plot_id = 'ep' + str(epoch)
        plot_training_progress(iter_ids=iter_ids, train_losses=train_losses,
                               iter_overall_iou_scores=iter_overall_iou_scores,
                               iter_per_class_iou_scores=iter_per_class_iou_scores,
                               show=False, save=True, save_dir=dir_checkpoint, id=loss_plot_id)

    plot_training_progress(iter_ids=iter_ids, train_losses=train_losses,
                           iter_overall_iou_scores=iter_overall_iou_scores,
                           iter_per_class_iou_scores=iter_per_class_iou_scores,
                           show=False, save=True, save_dir=dir_checkpoint, id=None)


if __name__ == '__main__':
    # args = get_args()
    args = cfg.Config()
    print('args : ', vars(args))

    str_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.dir_checkpoint = os.path.join(args.dir_checkpoint, args.cnn_arch, str_date_time)

    Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)

    validate_config(args)
    print("Configuration is valid.")

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger()

    # File output
    log_file = os.path.join(args.dir_checkpoint, 'training_log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.cnn_arch.lower() == 'unet':
        print('Using UNet ...')
        model = UNet(n_channels=args.number_of_in_channels, n_classes=args.classes, bilinear=args.bilinear)
    elif args.cnn_arch.lower() == 'deeplabv3+':
        print('Using DeepLab ...')
        assert args.backbone.lower() in ['resnet18', 'resnet50',
                                         'resnet101'], 'backbone must be one of resnet18, resnet50, resnet101 when using DeepLab'
        model = DeepLabv3Plus(n_channels=args.number_of_in_channels, n_classes=args.classes, os=16,
                              pretrained=args.use_pretrained_backbone, _print=True, backbone=args.backbone,
                              bilinear=args.bilinear)
    elif args.cnn_arch.lower() in ['fcn32s', 'fcn16s', 'fcn8s', 'fcns']:
        print('Using FCN ...')
        assert args.backbone in ['vgg11', 'vgg13', 'vgg16',
                                 'vgg19'], 'backbone must be one of vgg11, vgg13, vgg16, vgg19 when using FCN'

        model = FCN(n_channels=args.number_of_in_channels, n_classes=args.classes, backbone=args.backbone,
                    pretrained=args.use_pretrained_backbone, bilinear=args.bilinear, arch_type=args.cnn_arch,
                    requires_grad=True)
    elif args.cnn_arch.lower() == 'pspnet':
        print('Using PSPNet ...')
        assert args.backbone in ['resnet18', 'resnet50',
                                 'resnet101'], 'backbone must be one of resnet18, resnet50, resnet101 when using PSPNet'
        model = PSPNet(n_channels=args.number_of_in_channels, n_classes=args.classes, os=16, backbone=args.backbone,
                       pretrained=args.use_pretrained_backbone, bilinear=args.bilinear)
    elif args.cnn_arch.lower() == 'segnet':
        print('Using SegNet ...')
        model = SegNet(n_channels=args.number_of_in_channels, n_classes=args.classes, bilinear=args.bilinear)
    else:
        raise NotImplementedError(
            'Please provide a valid model name: unet, deeplabv3+, fcn32s, fcn16s, fcn8s, fcns, pspnet, segnet')

    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    ## copy config file to the checkpoint directory
    copyfile('config.py', args.dir_checkpoint + '/config.py')

    try:
        print('Training the model ...')
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,

            val_frequency=args.val_frequency / 100,
            amp=args.amp,

            train_images_dir=args.train_images_dir,
            train_mask_dir=args.train_mask_dir,
            val_images_dir=args.val_images_dir,
            val_mask_dir=args.val_mask_dir,
            dir_checkpoint=args.dir_checkpoint,
            target_size=args.target_size,
            loss_type=args.loss_type,
            class_weights=args.class_weights,
            extra_weight_frequency=args.extra_weight_frequency / 100,
            cnn_arch=args.cnn_arch,
            backbone=args.backbone,
            optimization=args.optimization,
            lr_policy=args.lr_policy,
            lr_decay_step=args.lr_decay_step,
            CE_variation=args.CE_variation,
            log_file=log_file,
        )
    except Exception as e:
        print('Error: ', e)
        raise e
