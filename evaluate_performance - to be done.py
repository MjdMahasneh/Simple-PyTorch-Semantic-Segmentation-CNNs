import logging
import os
import torch
from pathlib import Path

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate_dice as evaluate_with_dice
from evaluate import evaluate_iou as evaluate_with_iou

from unet import UNet
from deeplab import DeepLabv3Plus

from utils.data_loading import BasicDataset



# def iou_score(output, target):
#     smooth = 1e-5 ## a small constant added to the numerator and denominator) is a common practice to prevent division by zero in cases where the intersection and union might be zero, leading to an undefined IoU value
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#         output = (output > 0.5).astype(int)
#     else:
#         output = (output > 0.5).astype(int)
#
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy().astype(int)
#
#     intersection = (output & target).sum()
#     union = (output | target).sum()
#
#     iou = (intersection + smooth) / (union + smooth)
#
#     return iou
#
#
#
#
#
# @torch.inference_mode()
# def evaluate_with_iou(net, dataloader, device, amp):
#     net.eval()
#     num_val_batches = len(dataloader)
#     total_iou = 0.0
#
#     # Initializing a list to store the IoU for each class over all batches
#     classwise_iou = [0.0] * net.n_classes
#
#     with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
#         for batch in tqdm(dataloader, total=num_val_batches, desc='IoU evaluation', unit='batch', leave=False):
#             image, mask_true = batch['image'], batch['mask']
#
#             image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
#             mask_true = mask_true.to(device=device, dtype=torch.long)
#
#             mask_pred = net(image)
#
#             batch_iou = 0.0  # IoU accumulator for the batch
#
#             for cls in range(net.n_classes):  # Now including the background
#                 mask_pred_cls = (mask_pred.argmax(dim=1) == cls).float()
#                 mask_true_cls = (mask_true == cls).float()
#
#                 ##todo: test with the following, to see if IoU is as expected:
#                 ## (DONE) 100% correct : fill true mask with 1 and pred mask with 1
#                 ## (DONE) 50% correct: fill half true and pred mask with 0 and another half with 1
#                 ## (DONE) 0% correct : fill true mask with 0 and pred mask with 0
#                 ## keep in mind smooth = 1e-5 is added to the numerator and denominator to prevent division by zero which might slightly affect the results
#                 # mask_pred_cls[0] *= 0 ## mind that we slice [0] since its batch wise, and each batch has 2 images
#                 # mask_true_cls[0] *= 0
#                 # mask_pred_cls[0] += 1
#                 # mask_true_cls[0] += 0
#                 #
#                 # mask_pred_cls[1] *= 0
#                 # mask_true_cls[1] *= 0
#                 # mask_pred_cls[1] += 1
#                 # mask_true_cls[1] += 0
#
#                 ## Plotting
#                 # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#                 #
#                 # ## Convert the tensors to numpy arrays
#                 # mask_pred_cls_np = mask_pred_cls[0].cpu().numpy().squeeze()  # Pick the first image
#                 # mask_true_cls_np = mask_true_cls[0].cpu().numpy().squeeze()  # Pick the first image
#                 #
#                 # ## Display predicted mask for the class
#                 # ax[0].imshow(mask_pred_cls_np, cmap='gray')
#                 # ax[0].set_title('Predicted Mask for Class {}'.format(cls))
#                 #
#                 # ## Display ground truth mask for the class
#                 # ax[1].imshow(mask_true_cls_np, cmap='gray')
#                 # ax[1].set_title('True Mask for Class {}'.format(cls))
#                 #
#                 # plt.show()
#
#
#                 iou_cls = iou_score(mask_pred_cls, mask_true_cls)
#
#                 batch_iou += iou_cls
#                 classwise_iou[cls] += iou_cls  # Adding to the respective class
#
#             batch_iou /= net.n_classes  # Average the IoU over all classes
#
#             total_iou += batch_iou
#
#     # Average classwise IoU over all batches
#     classwise_iou = [iou / max(num_val_batches, 1) for iou in classwise_iou]
#
#     return total_iou / max(num_val_batches, 1), classwise_iou


class Config:
    '''Configuration class for training
        Usage:
            args = Config()
            print(vars(args))
            print(args.epochs)
    '''
    def __init__(self):

        # Size of each input batch. Default is 1.
        self.batch_size = 1


        # Whether to use mixed precision during training. Default is False.
        # self.amp = False


        # Whether to use bilinear upsampling. Default is False.
        self.bilinear = False

        # Number of output classes. Default is 2.
        self.classes = 3 #2

        self.target_size = (512, 512) ## (height, width)

        ## initialize the paths
        # self.dir_root = Path('G:/Datasets/carvana-image-masking-challenge')
        self.dir_root = Path('G:/Datasets/MoA_STC_dataset_export-20230630_170319 - II/MoA_semantic_seg_subset-1.0')
        # self.dir_img = Path(os.path.join(self.dir_root, 'images'))
        # self.dir_mask = Path(os.path.join(self.dir_root, 'masks'))
        self.train_images_dir = Path(os.path.join(self.dir_root, 'train/images'))
        self.train_mask_dir = Path(os.path.join(self.dir_root, 'train/masks'))
        self.val_images_dir = Path(os.path.join(self.dir_root, 'val/images'))
        self.val_mask_dir = Path(os.path.join(self.dir_root, 'val/masks'))

        self.cnn_arch = 'DeepLab' ## 'UNet' or 'DeepLab'

        self.model = './checkpoints - temp - DeepLab/checkpoint_epoch1_150.pth'








if __name__ == '__main__':
    # classwise_iou = [0.0] * 3
    # print('classwise_iou:', classwise_iou)
    # raise Exception('stop here!!')

    # args = get_args()
    args = Config()
    print('args : ', vars(args))

    ## IoU can slightly vary with batch size, so we set it to 1 for evaluation see https://stackoverflow.com/questions/71629966/does-batch-size-matters-at-inference-for-a-sematic-segmentation-model and https://github.com/IvLabs/stagewise-knowledge-distillation/issues/12 for more info.
    assert args.batch_size == 1, 'Please set batch size to 1 for evaluation to ensure consistency. See: https://stackoverflow.com/questions/71629966/does-batch-size-matters-at-inference-for-a-sematic-segmentation-model and https://github.com/IvLabs/stagewise-knowledge-distillation/issues/12 for more info.'


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # model = model.to(memory_format=torch.channels_last)

    if args.cnn_arch == 'UNet':
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.cnn_arch == 'DeepLab':
        model = DeepLabv3Plus(n_channels=3, n_classes=args.classes, os=16, pretrained=True, _print=True, backbone='resnet50', bilinear=True) ## only bilinear=True is supported for now when using DeepLab
    else:
        raise NotImplementedError('Please provide a valid model name: unet or deeplab')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')


    state_dict = torch.load(args.model, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)
    model.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{args.cnn_arch} architecture\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')


    assert args.val_images_dir is not None and args.val_mask_dir is not None, 'Please provide the path to the images directory'


    # Create datasets
    val_dataset = BasicDataset(args.val_images_dir, args.val_mask_dir, mask_suffix='', target_size=args.target_size, stage='val')
    n_val = len(val_dataset)
    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)

    # Create data loaders
    # val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)



    logging.info(f'''Starting IoU evaluation:
        Batch size:      {args.batch_size}
        Validation size: {n_val}
        Device:          {device.type}
        ''')

    val_score = evaluate_with_dice(model, val_loader, device, amp=False)
    logging.info('Validation Dice score: {}'.format(val_score))


    val_score, classwise_scores = evaluate_with_iou(model, val_loader, device, amp=False)
    logging.info('Validation IoU score: {}'.format(val_score))
    for i, cls_iou in enumerate(classwise_scores):
        logging.info(f'Class {i} IoU score: {cls_iou}')


