from typing import Any, Dict, Tuple
# --------
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.amp import autocast     # used in VarifocalLoss
import torch
import numpy as np

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=True, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(..., N, 4) and box2(..., M, 4).
    
    Args:
        box1 (torch.Tensor): Bounding boxes, shape (..., N, 4).
        box2 (torch.Tensor): Bounding boxes, shape (..., M, 4).
        xywh (bool): If True, input boxes are in (x, y, w, h) format. If False, (x1, y1, x2, y2).
        GIoU (bool): Compute Generalized IoU if True.
        DIoU (bool): Compute Distance IoU if True.
        CIoU (bool): Compute Complete IoU if True (default).
        eps (float): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values of shape (..., N, M).
    """
    # Convert xywh to xyxy if needed
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    # Intersection area
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist squared
            if CIoU:
                v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU
    return iou

def bbox2dist(anchor_points, bbox, reg_max):
    """
    Transform bbox(xyxy) to dist(ltrb).

    Args:
        anchor_points (torch.Tensor): Anchor points, shape (h*w, 2).
        bbox (torch.Tensor): Bounding boxes in xyxy format, shape (..., 4).
        reg_max (int): Maximum regression value.

    Returns:
        (torch.Tensor): Distances in ltrb format, shape (..., 4).
    """
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) to (x1, y1, x2, y2).

    Args:
        x (torch.Tensor | np.ndarray): Input boxes in xywh format, shape (..., 4).

    Returns:
        (torch.Tensor | np.ndarray): Boxes in xyxy format, shape (..., 4).
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchors from features.

    Args:
        feats (List[torch.Tensor]): Feature maps from the model, each of shape (b, c, h, w).
        strides (List[int]): Strides of each feature map.
        grid_cell_offset (float): Offset for grid cells.

    Returns:
        (tuple): (anchor_points, stride_tensor)
            - anchor_points (torch.Tensor): Anchor points, shape (h*w*num_levels, 2).
            - stride_tensor (torch.Tensor): Stride values for each anchor, shape (h*w*num_levels, 1).
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, (stride, feat) in enumerate(zip(strides, feats)):
        _, _, h, w = feat.shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Transform distance(ltrb) to box(xywh or xyxy).

    Args:
        distance (torch.Tensor): Distance from anchor points, shape (..., 4).
        anchor_points (torch.Tensor): Anchor points, shape (..., 2).
        xywh (bool): If True, return boxes in xywh format. If False, xyxy.
        dim (int): Dimension along which to split the distance.

    Returns:
        (torch.Tensor): Bounding boxes in xywh or xyxy format, shape (..., 4).
    """
    lt, rb = distance.split(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

class TaskAlignedAssigner(torch.nn.Module):
    """
    A task-aligned assigner for object detection.

    Args:
        topk (int): Number of top candidates to consider.
        num_classes (int): Number of classes.
        alpha (float): Weight for classification score.
        beta (float): Weight for localization (IoU).
        eps (float): Small value to avoid division by zero.
    """
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted class scores, shape (b, num_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted boxes in xyxy format, shape (b, num_anchors, 4).
            anc_points (torch.Tensor): Anchor points, shape (num_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels, shape (b, max_num_obj, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes in xyxy format, shape (b, max_num_obj, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth, shape (b, max_num_obj, 1).

        Returns:
            (tuple): (pred_scores, target_bboxes, target_scores, fg_mask, target_gt_idx)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.long),
            )

        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self._select_topk_candidates(
            align_metric, mask_pos, mask_gt=mask_gt
        )

        # Assigned target
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Normalize
        align_metric *= mask_pos
        pos_overlaps = (overlaps * mask_pos).sum(-1) / (mask_pos.sum(-1) + self.eps)
        pos_overlaps = pos_overlaps.clamp_(0.0, 1.0)
        target_scores = target_scores * pos_overlaps.unsqueeze(-1)

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def _get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask for each ground truth."""
        mask_in_gts, valid_mask = self._select_candidates_in_gts(anc_points, gt_bboxes)
        scores = self._get_alignment_metric(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * valid_mask)
        return self._get_pos_mask_topk(scores, mask_gt)

    def _get_pos_mask_topk(self, align_metric, mask_gt):
        """Get positive mask based on topk alignment metric."""
        mask_topk = self._select_topk_candidates(align_metric, mask_gt=mask_gt)
        mask_topk = mask_topk * mask_gt
        mask_pos = mask_topk.sum(-1) > 0
        return mask_pos, align_metric, mask_topk

    def _select_candidates_in_gts(self, anc_points, gt_bboxes):
        """
        Select anchor points within ground truth boxes.

        Args:
            anc_points (torch.Tensor): Anchor points, shape (num_anchors, 2).
            gt_bboxes (torch.Tensor): Ground truth boxes in xyxy format, shape (b, max_num_obj, 4).

        Returns:
            (tuple): (mask_in_gts, valid_mask)
        """
        n_anchors = anc_points.size(0)
        bs, n_boxes = gt_bboxes.shape[:2]
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, -1)
        x1y1, x2y2 = (anc_points - lt).view(bs * n_boxes, n_anchors, 2), (rb - anc_points).view(bs * n_boxes, n_anchors, 2)
        mask_in_gts = (x1y1 > 0) & (x2y2 > 0)
        mask_in_gts = mask_in_gts.all(-1).view(bs, n_boxes, n_anchors)
        valid_mask = (gt_bboxes.sum(-1) > 0).view(bs, n_boxes, 1)
        return mask_in_gts, valid_mask

    def _get_alignment_metric(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts):
        """Compute alignment metric."""
        pd_scores = pd_scores.permute(0, 2, 1)
        na = pd_bboxes.size(1)
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        for i in range(self.bs):
            overlaps[i] = self._iou_calculation(gt_bboxes[i], pd_bboxes[i])[0]
        scores = pd_scores[torch.arange(self.bs, device=pd_scores.device)[:, None], gt_labels.squeeze(-1).long()]
        return scores.pow(self.alpha) * overlaps.pow(self.beta)

    def _iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1), None

    def _select_topk_candidates(self, align_metric, mask_gt, large_batch=False):
        """
        Select topk candidates based on alignment metric.

        Args:
            align_metric (torch.Tensor): Alignment metric, shape (b, max_num_obj, num_anchors).
            mask_gt (torch.Tensor): Mask for valid ground truth, shape (b, max_num_obj, 1).
            large_batch (bool): If True, use chunking for large batches.

        Returns:
            (tuple): (target_gt_idx, fg_mask, mask_pos)
        """
        n_anchors = align_metric.size(2)
        if large_batch:
            topk_mask = torch.zeros_like(align_metric, dtype=torch.bool)
            for i in range(self.bs):
                topk_mask[i] = torch.topk(align_metric[i], self.topk, dim=1, largest=True)[1]
        else:
            topk_mask = torch.topk(align_metric, self.topk, dim=2, largest=True)[1]
            topk_mask = torch.zeros([self.bs, self.n_max_boxes, n_anchors], dtype=torch.bool, device=align_metric.device).scatter_(2, topk_mask, 1)
        target_gt_idx = topk_mask.argmax(1)
        fg_mask = topk_mask.sum(-1) > 0
        return target_gt_idx, fg_mask, topk_mask

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Get target labels, boxes, and scores.

        Args:
            gt_labels (torch.Tensor): Ground truth labels, shape (b, max_num_obj, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes, shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of assigned ground truth, shape (b, num_anchors).
            fg_mask (torch.Tensor): Foreground mask, shape (b, num_anchors).

        Returns:
            (tuple): (target_labels, target_bboxes, target_scores)
        """
        batch_idx = torch.arange(self.bs, device=gt_labels.device)[:, None]
        target_labels = gt_labels[batch_idx, target_gt_idx]
        target_bboxes = gt_bboxes[batch_idx, target_gt_idx]
        target_scores = torch.zeros((self.bs, fg_mask.size(1), self.num_classes), dtype=gt_labels.dtype, device=gt_labels.device)
        target_scores[fg_mask, target_labels.squeeze(-1)] = 1.0
        target_labels[fg_mask == 0] = self.bg_idx
        return target_labels.squeeze(-1), target_bboxes, target_scores
    
# class VarifocalLoss(nn.Module): #
#     """
#     Varifocal loss by Zhang et al.

#     Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
#     hard-to-classify examples and balancing positive/negative samples.

#     Attributes:
#         gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
#         alpha (float): The balancing factor used to address class imbalance.

#     References:
#         https://arxiv.org/abs/2008.13367
#     """

#     def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
#         """Initialize the VarifocalLoss class with focusing and balancing parameters."""
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha

#     def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
#         """Compute varifocal loss between predictions and ground truth."""
#         weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
#         with autocast(enabled=False):
#             loss = (
#                 (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
#                 .mean(1)
#                 .sum()
#             )
#         return loss


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

from types import SimpleNamespace

# ... (other helper functions and classes: bbox_iou, make_anchors, TaskAlignedAssigner, BboxLoss, etc.) ...

class v8DetectionLoss:
    """Criterion class for YOLOv8 object‐detection losses."""

    def __init__(self, model, tal_topk: int = 10):
        device = next(model.parameters()).device
        # Wrap hyperparameters dict so we can do self.hyp.box, etc.
        self.hyp = SimpleNamespace(**model.args)

        m = model.model[-1]  # The Detect() head
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(
        self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor
    ) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)
        idx = targets[:, 0]
        _, counts = idx.unique(return_counts=True)
        counts = counts.to(torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        for i in range(batch_size):
            mask = idx == i
            if mask.sum():
                out[i, : mask.sum()] = targets[mask, 1:]
        # xywh→xyxy and scale
        out[..., 1:5] = xywh2xyxy(out[..., 1:5] * scale_tensor)
        return out

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor
    ) -> torch.Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(
        self, preds: Any, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # preds → list of feature outputs
        feats = preds[1] if isinstance(preds, tuple) else preds
        combined = torch.cat(
            [f.view(feats[0].shape[0], self.no, -1) for f in feats], dim=2
        )
        pred_dist, pred_scores = combined.split(
            (self.reg_max * 4, self.nc), dim=1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()

        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype

        # anchors + strides
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_pts, stride_tensor = make_anchors(feats, self.stride, grid_cell_offset=0.5)

        # Build flat targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1),
             batch["cls"].view(-1, 1),
             batch["bboxes"]),
            dim=1,
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True) > 0

        # decode predicted boxes
        pred_bboxes = self.bbox_decode(anchor_pts, pred_dist)

        # assign
        _, tgt_bboxes, tgt_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_pts * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        denom = max(tgt_scores.sum(), 1)

        # classification loss
        loss_cls = (
            self.bce(pred_scores, tgt_scores.to(dtype))
            .sum() / denom
        )

        # box + dfl losses
        if fg_mask.sum():
            tgt_bboxes /= stride_tensor
            loss_iou, loss_dfl = self.bbox_loss(
                pred_dist, pred_bboxes, anchor_pts,
                tgt_bboxes, tgt_scores, denom, fg_mask
            )
        else:
            loss_iou = torch.tensor(0.0, device=self.device)
            loss_dfl = torch.tensor(0.0, device=self.device)

        # apply gains
        loss_iou *= self.hyp.box
        loss_cls *= self.hyp.cls
        loss_dfl *= self.hyp.dfl

        total = (loss_iou + loss_cls + loss_dfl) * batch_size
        return total, torch.stack([loss_iou, loss_cls, loss_dfl])