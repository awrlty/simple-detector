import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

import config
from utils import intersection_over_union


class CornerLoss(nn.Module):
    def __init__(self):
        super(CornerLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = config.S
        self.B = config.B  # number of boxes to predict
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, pred_tensor, target_tensor):  # input tensor: (batch_size, S, S, 10)
        iou_b1 = intersection_over_union(pred_tensor[..., :4], target_tensor[..., :4])  # (batch_size, S, S, 1)
        iou_b2 = intersection_over_union(pred_tensor[..., 5:9], target_tensor[..., :4])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)  # (2, batch_size, S, S, 1)

        # bestbox: indicies of 0, 1 for the best bbox
        _, bestbox = torch.max(ious, dim=0)  # (batch_size, S, S, 1)
        exists_box = target_tensor[..., 4].unsqueeze(-1)  # Binary mask(obj existence) Iobj_i: (batch_size, S, S, 1)

        # Loss for Coordinates
        box_predictions = exists_box * (bestbox * pred_tensor[..., 5:9] + (1 - bestbox) * pred_tensor[..., :4])
        box_targets = exists_box * target_tensor[..., :4]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + config.EPSILON))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # Loss for Object confidence
        pred_box = (bestbox * pred_tensor[..., 9:10] + (1 - bestbox) * pred_tensor[..., 4:5])
        obj_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target_tensor[..., 4:5]),
        )

        # Loss for no object
        no_obj_loss = self.mse(
            torch.flatten((1 - exists_box) * pred_tensor[..., 4:5], start_dim=1),
            torch.flatten((1 - exists_box) * target_tensor[..., 4:5], start_dim=1)
        )
        no_obj_loss += self.mse(
            torch.flatten((1 - exists_box) * pred_tensor[..., 9:10], start_dim=1),
            torch.flatten((1 - exists_box) * target_tensor[..., 4:5], start_dim=1)
        )

        loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * no_obj_loss
        )
        return loss


def test_iou():
    bbox1 = torch.tensor([[884, 512, 904, 532]])
    bbox2 = torch.tensor([[870, 502, 890, 522]])

    # iou = intersection_over_union(bbox2, bbox1, box_format="corners")
    # print("self-calculated:", iou)

    # iou2 = box_iou(bbox1, bbox2)
    # print("torch-calculated:", iou2)


def test_loss():
    from torch.utils.data import DataLoader
    from model import CornerDetectionNet
    from dataset import PointDataset

    trainset = PointDataset('train', augment=False)
    train_loader = DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CornerDetectionNet().to(device)
    criterion = CornerLoss()

    model.train()
    for i, (images, targets) in enumerate(train_loader):
        # if i == 2:
        images, targets = images.to(device), targets.to(device)  # targets = (batch_size, S, S, 10)
        # print(targets.sum())
        preds = model(images)
        # print(preds.sum())
        # print(torch.isnan(preds).int().sum().item())  # nan check

        loss = criterion(preds, targets)
        print(loss)


if __name__ == "__main__":
    # test_iou()
    test_loss()
