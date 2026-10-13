"""Text-aware classification for torchvision Mask R-CNN.

For every RoI the OCR label tokens whose centre lies inside the proposal box
are pooled (confidence-weighted mean of their features, max of their
class-name similarities, a has-text flag and log token count). An MLP maps
this to the 1024-d box representation and the result is *added* to the box
feature that feeds ``cls_score`` only; box regression and the mask branch are
unchanged. The MLP's last layer starts at zero, so at initialisation the
model is exactly the plain Mask R-CNN.

Tokens travel next to the images: the training/inference loop calls
``set_text_tokens`` before each forward, with token boxes normalised to the
original image (0..1). Proposals are normalised by the resized image sizes
that torchvision's transform records, so no coordinate bookkeeping through the
resize is needed.
"""
import torch
from torch import nn


class TextContext:
    def __init__(self):
        self.proposals = None
        self.image_shapes = None
        self.tokens = None


class ProposalRecorder(nn.Module):
    """Wraps ``box_roi_pool`` to remember the proposals it pooled."""

    def __init__(self, pool, context):
        super().__init__()
        self.pool = pool
        self.context = context

    def forward(self, features, proposals, image_shapes):
        self.context.proposals = proposals
        self.context.image_shapes = image_shapes
        return self.pool(features, proposals, image_shapes)


def roi_text_features(proposals, image_shapes, tokens, sim_slice, min_inside=None):
    """[sum R_i, D + S + 2] pooled token features for all proposals.

    A token belongs to a proposal if its box centre lies inside it (default) or,
    with ``min_inside``, if at least that fraction of the token box lies inside.
    """
    rows = []
    for boxes, shape, (token_boxes, feats, conf) in zip(proposals, image_shapes, tokens):
        height, width = float(shape[0]), float(shape[1])
        scale = boxes.new_tensor([width, height, width, height])
        norm = boxes / scale
        n_rois, dim = boxes.shape[0], feats.shape[1]
        n_sim = sim_slice[1] - sim_slice[0]
        if token_boxes.shape[0] == 0 or n_rois == 0:
            rows.append(boxes.new_zeros((n_rois, dim + n_sim + 2)))
            continue
        if min_inside is None:
            cx = (token_boxes[:, 0] + token_boxes[:, 2]) / 2
            cy = (token_boxes[:, 1] + token_boxes[:, 3]) / 2
            inside = ((cx[None] >= norm[:, 0:1]) & (cx[None] <= norm[:, 2:3])
                      & (cy[None] >= norm[:, 1:2]) & (cy[None] <= norm[:, 3:4]))
        else:
            tb = token_boxes.to(norm.dtype)
            ix = (torch.minimum(norm[:, None, 2], tb[None, :, 2]) - torch.maximum(norm[:, None, 0], tb[None, :, 0])).clamp(min=0)
            iy = (torch.minimum(norm[:, None, 3], tb[None, :, 3]) - torch.maximum(norm[:, None, 1], tb[None, :, 1])).clamp(min=0)
            area = ((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])).clamp(min=1e-12)
            inside = (ix * iy) / area[None] >= float(min_inside)
        weights = inside.to(feats.dtype) * conf[None].to(feats.dtype)
        total = weights.sum(1, keepdim=True)
        mean = (weights @ feats) / total.clamp(min=1e-6)
        sims = feats[:, sim_slice[0]:sim_slice[1]]
        masked = sims[None].expand(n_rois, -1, -1).masked_fill(~inside[:, :, None], float("-inf"))
        has = inside.any(1)
        max_sim = torch.where(has[:, None], masked.max(1).values, torch.zeros_like(masked[:, 0]))
        count = torch.log1p(inside.sum(1).to(feats.dtype))[:, None] / 3.0
        rows.append(torch.cat([mean, max_sim, has[:, None].to(feats.dtype), count], dim=1).to(boxes.dtype))
    return torch.cat(rows, dim=0)


class TextAwarePredictor(nn.Module):
    def __init__(self, base, context, text_dim, sim_slice, hidden=256, token_dropout=0.15, image_dropout=0.15,
                 min_inside=None):
        super().__init__()
        self.cls_score = base.cls_score
        self.bbox_pred = base.bbox_pred
        self.context = context
        self.sim_slice = (int(sim_slice[0]), int(sim_slice[1]))
        representation = base.cls_score.in_features
        in_dim = int(text_dim) + (self.sim_slice[1] - self.sim_slice[0]) + 2
        self.text_mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                      nn.Linear(hidden, representation))
        nn.init.zeros_(self.text_mlp[2].weight)
        nn.init.zeros_(self.text_mlp[2].bias)
        self.token_dropout = float(token_dropout)
        self.image_dropout = float(image_dropout)
        self.min_inside = None if min_inside is None else float(min_inside)

    def _tokens(self):
        tokens = self.context.tokens
        if tokens is None:
            raise RuntimeError("Text-aware model called without set_text_tokens()")
        self.context.tokens = None
        if not self.training or (self.token_dropout == 0 and self.image_dropout == 0):
            return tokens
        dropped = []
        for boxes, feats, conf in tokens:
            keep = torch.rand(boxes.shape[0], device=boxes.device) >= self.token_dropout
            if torch.rand(()) < self.image_dropout:
                keep = torch.zeros_like(keep)
            dropped.append((boxes[keep], feats[keep], conf[keep]))
        return dropped

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        text = roi_text_features(self.context.proposals, self.context.image_shapes, self._tokens(),
                                 self.sim_slice, self.min_inside)
        if text.shape[0] != x.shape[0]:
            raise RuntimeError(f"RoI count mismatch: {text.shape[0]} text rows vs {x.shape[0]} features")
        scores = self.cls_score(x + self.text_mlp(text.to(x.dtype)))
        return scores, self.bbox_pred(x)


def attach_text_head(model, text_head_cfg):
    """Convert a built Mask R-CNN in place; returns the model."""
    context = TextContext()
    heads = model.roi_heads
    heads.box_roi_pool = ProposalRecorder(heads.box_roi_pool, context)
    heads.box_predictor = TextAwarePredictor(
        heads.box_predictor, context, text_head_cfg["text_dim"], text_head_cfg["sim_slice"],
        text_head_cfg.get("hidden", 256), text_head_cfg.get("token_dropout", 0.15),
        text_head_cfg.get("image_dropout", 0.15), text_head_cfg.get("min_inside"))
    model.text_context = context
    return model


def is_text_aware(model):
    return getattr(model, "text_context", None) is not None


def set_text_tokens(model, targets, device):
    """Hand the OCR tokens of a batch (from dataset targets) to the model."""
    tokens = []
    for target in targets:
        if "_ocr_boxes" not in target:
            raise KeyError("Text-aware model requires datasets built with an OCR provider")
        tokens.append((target["_ocr_boxes"].to(device), target["_ocr_feats"].to(device),
                       target["_ocr_conf"].to(device)))
    model.text_context.tokens = tokens


def text_parameters(model):
    """Names and parameters of the text branch (for a separate optimizer)."""
    return [(name, param) for name, param in model.named_parameters()
            if ".text_mlp." in name and param.requires_grad]
