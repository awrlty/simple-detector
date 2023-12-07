from collections import Counter

import numpy as np

import torch


def mean_average_precidion(pred_boxes, target_boxes, iou_thresh=0.5, box_format="midpoint"):
    """
    Args:
        pred_boxes: (list) [[x, y, w, h], [...], ...]
        target_boxes: (list) [[x, y, w, h], [...], ...]
        iou_thresh: (float)
        box_format: (str) "midpoint" or "corners"

    Returns:
        (float) mAP value given a specific IoU threshold
    """
    average_precisions = []
    num_bboxes = Counter([gt[0] for gt in target_boxes])
    print(num_bboxes)


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from dataset import PointDataset
    from model import CornerDetectionNet
    import config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CornerDetectionNet().to(device)

    trainset = PointDataset("train", augment=True)
    train_loader = DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=config.NUM_WORKERS)

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images)

        map = mean_average_precidion(preds, targets, iou_thresh=0.3)
