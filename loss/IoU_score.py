import torch
from torch import Tensor


def iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of IoU coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    
    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2*inter

        return (inter + epsilon) / (sets_sum - inter + epsilon)
    else:
        # compute and average metric for each batch element
        iou = 0
        objs = 0
        for i in range(input.shape[0]):
            is_mask = torch.sum(target[i, ...])
            if is_mask != 0:
                iou += iou_coeff(input[i, ...], target[i, ...])
                objs += 1
        return iou / objs


def multiclass_iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of IoU coefficient for all classes
    assert input.size() == target.size()
    iou = 0
    objs = 0
    for channel in range(input.shape[1]):
        is_mask = torch.sum(target[:, channel, ...])
        if is_mask != 0:
            iou += iou_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
            objs += 1
    return iou / objs


def iou_loss(input: Tensor, target: Tensor, multiclass: bool = True):
    # IoU loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_iou_coeff if multiclass else iou_coeff
    return 1 - fn(input, target, reduce_batch_first=True)