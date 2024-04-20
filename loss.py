import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import intersection_over_union

class YoloV1Loss(nn.Module):
    def __init__(self, S, B, C, coord_lambda, noobj_lambda, device):
        super().__init__()

        self.mse = nn.MSELoss(reduction='sum')

        self.S = S
        self.B = B
        self.C = C
        self.coord_lambda = coord_lambda
        self.noobj_lambda = noobj_lambda
        self.device = device

    def compute_iou(self, bbox1, bbox2):
        """
        Compute intersection over union for preds box and target box. Box format [x1, y1, x2, y2]
        Args:
            (Tensor): bbox1: [N, 4]
            (Tensor): bbox2: [M, 4]
        Returns:
            (Tensor): iou: [B, 1]
        """

        N = bbox1.size(0)
        M = bbox2.size(0)

        # Left top coordinate of intersection
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand_as(N, M, 2), # [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand_as(N, M, 2), # [N, M, 2]
        )

        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand_as(N, M, 2), # [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand_as(N, M, 2), # [N, M, 2]
        )

        # Compute weight and height in intersection
        wh = rb - lt
        wh[wh<0] = 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of bbox
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])

        area1 = area1.unsqueeze(1).expand_as(inter)
        area2 = area1.unsqueeze(0).expand_as(inter)

        union  = area1 + area2 - inter # [N, M, 2]
        iou = inter / union # [N, M, 2]

        return iou

    def forward(self, preds, targets):
        """
        Compute loss for yolo training 
        Args:
            (Tensor): preds: [batch_size, S, S, 2*B + C]
            (Tensor): targets: [batch_size, S, S, 2*B + C]
        Returns:
            (Tensor): loss: [1]
        """
        S, B, C = self.S, self.B, self.C
        N = 2 * B + C

        batch_size = targets.size(0)

        # Create object mark
        obj_mark = targets[:, :, :, 4] > 0 # [batch_size, S, S]
        obj_mark = obj_mark.unsqueeze(-1).expand_as(targets) # [batch_size, S, S] => [batch_size, S, S, N]
        
        # Create non object mark
        no_obj_mark = targets[:, :, :, 4] == 0 # [batch_size, S, S]
        no_obj_mark = no_obj_mark.unsqueeze(-1).expand_as(targets)  # [batch_size, S, S] => [batch_size, S, S, N]

        # Get cell that contain object
        obj_targets = targets[obj_mark].view(-1, N) # [n_obj, N]: n_obj is number of cells which contain objects.
        bbox_targets = obj_targets[:, :5*B].contiguous.view(-1, 5) # [n_obj*B, 5]
        class_targets = obj_targets[:, 5*B:] # [n_obj, C]

        obj_preds = preds[obj_mark].view(-1, N) # [n_obj, N]: n_obj is number of cells which contain objects.
        bbox_preds = obj_preds[:, :5*B].contiguous.view(-1, 5) # [n_obj*B, 5]
        class_preds = obj_preds[:, 5*B:] # [n_obj, C]

        # Get cell that do not contrain object
        no_obj_targets = targets[no_obj_mark].view(-1, N) # [n_obj, N]: n_obj is number of cells which do not contain objects.
        no_obj_preds = preds[no_obj_mark].view(-1, N) # [n_obj, N]: n_obj is number of cells which do not contain objects.


        # ==================================
        # ==        NON OBJECT LOSS       ==
        # ==================================
        non_obj_mark_conf = torch.ByteTensor(no_obj_preds.size()).fill_(0).to(self.device) # [n_obj, N]
        for b in range(B):
            non_obj_mark_conf[:, 4 + b*5] = 1 # non_obj_mark_coord[:, 4] = 1, non_obj_mark_coord[:, 9] = 1
        no_obj_targets_conf = no_obj_targets[non_obj_mark_conf] # [n_obj, 2=[non object box 1, non object box 2]]
        no_obj_preds_conf = no_obj_preds[non_obj_mark_conf] # [n_obj, 2=[non object box 1, non object box 2]]
        loss_noobj = F.mse_loss(no_obj_targets_conf, no_obj_preds_conf, reduction='sum')


        # ==================================
        # ==          OBJECT LOSS         ==
        # ==================================
        coord_response_mask = torch.ByteTensor(bbox_targets.size()).fill_(0).to(self.device)     # [n_obj*B, 5]
        coord_not_response_mask = torch.ByteTensor(bbox_targets.size()).fill_(1).to(self.device) # [n_obj*B, 5]
        bbox_target_iou = torch.zeros(bbox_targets.size(), device=self.device) # [n_obj*B, 5]
        for i in range(0, bbox_targets.size(0), B):
            preds = bbox_preds[i:i+B] # [B, 5]
            preds_xyxy = torch.FloatTensor(preds.size())
            preds_xyxy[:, :2] = preds_xyxy[:, :2]/float(S) - 0.5 * preds[:, 2:4]
            preds_xyxy[:, 2:4] = preds_xyxy[:, :2]/float(S) + 0.5 * preds[:, 2:4]

            targets = bbox_targets[i].view(-1, 5) # [1, 5]
            targets_xyxy = torch.FloatTensor(targets.size())
            targets_xyxy[:, :2] = targets_xyxy[:, :2]/float(S) - 0.5 * targets[:, 2:4]
            targets_xyxy[:, 2:4] = targets_xyxy[:, :2]/float(S) + 0.5 * targets[:, 2:4]

            iou = self.compute_iou(preds_xyxy[:, :4], targets_xyxy[:, :4])
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.to(self.device)

            coord_response_mask[i+max_index] = 1
            coord_not_response_mask[i+max_index] = 0

            bbox_target_iou[i+max_index, torch.LongTensor([4]).to(self.device)] = (max_iou).data.to(self.device)

        bbox_pred_response = bbox_preds[coord_response_mask].view(-1, 5)
        bbox_target_response = bbox_targets[coord_response_mask].view(-1, 5)
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_target_response[:, 4], target_iou[:, 4], reduction='sum')

        loss_class = F.mse_loss(class_preds, class_targets, reduction='sum')

        loss = self.coord_lambda * (loss_xy + loss_wh) + loss_obj + self.noobj_lambda * loss_noobj + loss_class
        loss = loss / batch_size

        return loss