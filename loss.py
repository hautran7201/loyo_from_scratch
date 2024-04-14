import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloV1Loss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mse = nn.MSELoss(reduction='sum')

        self.B = config.B
        self.C = config.C
        self.depth = 5*self.B + self.C
        self.coord_lambda = config.coord_lambda
        self.noobj_lambda = config.noobj_lambda

    def forward(self, preds, targets):
        preds_shape = preds.shape

        # (Batch, S, S, Depth)
        grid_shape = int((preds_shape[-1]//self.depth)**(0.5))
        preds = preds.reshape(-1, grid_shape, grid_shape, self.depth)
        
        iou_b1 = intersection_over_union(preds[..., 21:25], targets[..., 21:25])
        iou_b2 = intersection_over_union(preds[..., 26:30], targets[..., 21:25])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = targets[..., 20].unsqueeze(3)

        # Box predictions 
        box_predictions = exists_box * (
              (
                  bestbox * preds[..., 26:30]
                  + (1-bestbox) * preds[..., 21:25]
              )
        )

        box_targets = exists_box * targets[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])      

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # Object loss
        pred_obj = (
            bestbox * preds[..., 25:26] + (1-bestbox) * preds[..., 20:21]
        )

        obj_loss = self.mse(
            torch.flatten(exists_box * pred_obj),
            torch.flatten(exists_box * targets[..., 20:21])
        )

        # Non object loss
        non_obj_loss = self.mse(
            torch.flatten((1-exists_box)*box_predictions[..., 20:21]),
            torch.flatten((1-exists_box)*box_targets[..., 20:21])
        )

        non_obj_loss += self.mse(
            torch.flatten((1-exists_box)*box_predictions[..., 26:30]),
            torch.flatten((1-exists_box)*box_targets[..., 20:21])
        )

        # Class loss
        class_loss = self.mse(
            torch.flatten(exists_box * preds[..., :20], end_dim=-2),
            torch.flatten(exists_box * targets[..., :20], end_dim=-2)
        )

        loss = (
            self.coord_lambda * box_loss  # first two rows in paper
            + obj_loss  # third row in paper
            + self.noobj_lambda * non_obj_loss  # forth row
            + class_loss  # fifth row
        )

        return loss