import torch

def iou_score(output, target):
    smooth = 1e-5 ## a small constant added to the numerator and denominator) is a common practice to prevent division by zero in cases where the intersection and union might be zero

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
        output = (output > 0.5).astype(int)
    else:
        output = (output > 0.5).astype(int)

    if torch.is_tensor(target):
        target = target.data.cpu().numpy().astype(int)

    intersection = (output & target).sum()
    union = (output | target).sum()

    iou = (intersection + smooth) / (union + smooth)

    return iou
