import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from src.models.mask_rcnn import build_mask_rcnn


@pytest.mark.slow
def test_mask_rcnn_forward_pass():
    model = build_mask_rcnn(20, min_size=64, max_size=128, pretrained=False).eval()
    with torch.inference_mode():
        output = model([torch.rand(3, 96, 128)])[0]
    assert {"boxes", "labels", "scores", "masks"}.issubset(output)

