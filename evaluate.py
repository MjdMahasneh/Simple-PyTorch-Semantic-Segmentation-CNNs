import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.iou_score import iou_score


@torch.inference_mode()
def evaluate_iou(net, dataloader, device, amp):
    '''
    Please set batch size to 1 for evaluation to ensure consistency.
       See: https://stackoverflow.com/questions/71629966/does-batch-size-matters-at-inference-for-a-sematic-segmentation-model
       and https://github.com/IvLabs/stagewise-knowledge-distillation/issues/12 for more info.
    '''
    net.eval()
    num_val_batches = len(dataloader)
    total_iou = 0.0

    # Initializing a list to store the IoU for each class over all batches
    classwise_iou = [0.0] * net.n_classes

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='IoU evaluation', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            batch_iou = 0.0  # IoU accumulator for the batch

            for cls in range(net.n_classes):  # Now including the background
                mask_pred_cls = (mask_pred.argmax(dim=1) == cls).float()
                mask_true_cls = (mask_true == cls).float()

                iou_cls = iou_score(mask_pred_cls, mask_true_cls)

                batch_iou += iou_cls
                classwise_iou[cls] += iou_cls  # Adding to the respective class

            batch_iou /= net.n_classes  # Average the IoU over all classes

            total_iou += batch_iou

    # Average classwise IoU over all batches
    classwise_iou = [iou / max(num_val_batches, 1) for iou in classwise_iou]

    return total_iou / max(num_val_batches, 1), classwise_iou


@torch.inference_mode()
def evaluate_dice(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
