import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    """
    Calculate the Intersection over Union (IoU) for predictions.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.
        n_classes (int, optional): Number of classes. Default is 21.

    Returns:
        float: Mean IoU across all classes.
    """

    # Calculate IoU on a per-class basis
    iou_list = torch.empty(n_classes)
    for c in range(n_classes):
        tp = ((pred == c) & (target == c)).sum()
        fp = ((pred == c) & (target != c)).sum()
        fn = ((pred != c) & (target == c)).sum()
        iou_list[c] = tp/(tp+fp+fn)

    # Ignore NaN when averaging IoU values
    return torch.nanmean(iou_list)

def pixel_acc(pred, target):
    """
    Calculate pixel-wise accuracy between predictions and targets.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.

    Returns:
        float: Pixel-wise accuracy.
    """
    correct = (pred == target).sum()
    total = pred.numel()
    return correct/total