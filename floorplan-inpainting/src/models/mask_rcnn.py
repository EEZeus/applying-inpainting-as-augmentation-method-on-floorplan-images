from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_mask_rcnn(num_classes=20, min_size=1024, max_size=1600, pretrained=True, text_head=None):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    kwargs = {} if pretrained else {"weights_backbone": None}
    model = maskrcnn_resnet50_fpn_v2(weights=weights, min_size=min_size, max_size=max_size, **kwargs)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    if text_head:
        from .text_head import attach_text_head

        attach_text_head(model, text_head)
    return model
