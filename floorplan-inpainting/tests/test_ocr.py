import json

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from src.augmentation.text_relabel import (apply_patch, convert_room, find_font, make_patch, render_word,
                                           room_label_words, verify_reading)
from src.data.transforms import DetectionTransform
from src.experiments.train import _validate_experiment
from src.models.mask_rcnn import build_mask_rcnn
from src.models.text_head import roi_text_features, set_text_tokens
from src.ocr.features import OcrProvider, build_text_features, label_texts, save_features
from src.ocr.text import (char_ngram_hash, is_label_text, merge_tile_tokens, normalize_text, text_similarity,
                          tile_origins, tokens_in_mask)
from src.utils.config import load_yaml, resolve_path


def test_normalisation_drops_room_numbers_and_keeps_words():
    assert normalize_text("Büro 45-03-012") == "büro"
    assert normalize_text("Spindraum H") == "spindraum h"
    assert not is_label_text("45-03-012")
    assert not is_label_text("F90")
    assert is_label_text("WC")
    assert text_similarity("Buero", "Büro") > 0.75
    assert text_similarity("Abstellraum", "Büro") < 0.5
    vector = char_ngram_hash("Putzr.")
    assert vector.shape == (256,) and abs(np.linalg.norm(vector) - 1) < 1e-6
    assert np.allclose(vector, char_ngram_hash("putzr"))


def test_tiles_cover_and_duplicates_merge():
    starts = tile_origins(5000, 1600, 256)
    assert starts[0] == 0 and starts[-1] + 1600 == 5000
    assert all(b - a <= 1600 - 256 for a, b in zip(starts, starts[1:]))
    tokens = [{"box": [10, 10, 60, 30], "text": "Büro", "conf": 0.9},
              {"box": [12, 10, 40, 30], "text": "Bü", "conf": 0.5},
              {"box": [200, 10, 260, 30], "text": "WC", "conf": 0.8}]
    merged = merge_tile_tokens(tokens)
    assert [t["text"] for t in merged] == ["Büro", "WC"]


def test_tokens_in_mask_uses_box_centres():
    mask = np.zeros((100, 100), bool); mask[:50, :50] = True
    assert tokens_in_mask([[10, 10, 20, 20], [40, 40, 80, 80], [60, 60, 70, 70]], mask).tolist() == [0]


def test_roi_text_features_pools_tokens_inside_each_proposal():
    proposals = [torch.tensor([[0., 0., 50., 50.], [50., 50., 100., 100.], [0., 60., 10., 70.]])]
    shapes = [(100, 100)]
    feats = torch.tensor([[1., 0., 0.9, 0.1], [0., 1., 0.2, 0.7]])
    boxes = torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.6, 0.6, 0.7, 0.7]])
    conf = torch.tensor([1.0, 0.5])
    out = roi_text_features(proposals, shapes, [(boxes, feats, conf)], (2, 4))
    assert out.shape == (3, 4 + 2 + 2)
    assert torch.allclose(out[0, :4], feats[0]) and torch.allclose(out[1, :4], feats[1])
    assert torch.allclose(out[0, 4:6], feats[0, 2:4])
    assert out[0, 6] == 1 and out[2, 6] == 0 and torch.all(out[2] == 0)


def _tiny_model(text):
    head = {"text_dim": 8, "sim_slice": [6, 8], "hidden": 16} if text else None
    torch.manual_seed(0)
    return build_mask_rcnn(20, 128, 160, pretrained=False, text_head=head)


def _batch():
    image = torch.rand(3, 128, 160)
    mask = torch.zeros(1, 128, 160, dtype=torch.uint8); mask[0, 20:80, 30:100] = 1
    target = {"boxes": torch.tensor([[30., 20., 100., 80.]]), "labels": torch.tensor([12]), "masks": mask,
              "_ocr_boxes": torch.tensor([[0.3, 0.3, 0.4, 0.35]]), "_ocr_feats": torch.rand(1, 8),
              "_ocr_conf": torch.tensor([0.9])}
    return [image], [target]


def test_text_aware_model_trains_and_starts_identical_to_plain_model():
    plain, text = _tiny_model(False), _tiny_model(True)
    missing, unexpected = text.load_state_dict(plain.state_dict(), strict=False)
    assert all(key.startswith("roi_heads.box_predictor.text_mlp") for key in missing) and not unexpected
    images, targets = _batch()
    for model in (plain, text):
        model.eval()
    with torch.no_grad():
        set_text_tokens(text, targets, "cpu")
        out_text = text(images)
        out_plain = plain(images)
    assert len(out_text[0]["scores"]) == len(out_plain[0]["scores"])
    assert torch.allclose(out_text[0]["scores"], out_plain[0]["scores"], atol=1e-5)
    text.train()
    set_text_tokens(text, targets, "cpu")
    losses = text(images, [{k: v for k, v in t.items() if not k.startswith("_")} for t in targets])
    sum(losses.values()).backward()
    assert text.roi_heads.box_predictor.text_mlp[2].weight.grad is not None
    with pytest.raises(RuntimeError):
        text.eval()(images)  # tokens are consumed per forward


def test_flip_mirrors_ocr_boxes():
    image = Image.new("RGB", (100, 50), "white")
    target = {"boxes": torch.tensor([[10., 10., 30., 20.]]), "masks": torch.zeros(1, 50, 100, dtype=torch.uint8),
              "_ocr_boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]])}
    _, out = DetectionTransform(train=True, flip_probability=1.0)(image, target)
    assert torch.allclose(out["_ocr_boxes"], torch.tensor([[0.7, 0.2, 0.9, 0.4]]))


class _FakeEncoder:
    model_name = "fake"

    def encode(self, texts):
        rows = [np.array([len(t), t.count("a"), 1.0], dtype=np.float32) for t in texts]
        rows = [r / np.linalg.norm(r) for r in rows]
        return np.stack(rows) if rows else np.zeros((0, 3), np.float32)


def test_provider_returns_label_tokens_normalised(tmp_path):
    cache = {"metadata": {}, "images": {"p1.png": {"width": 200, "height": 100, "tokens": [
        {"box": [20, 10, 60, 30], "text": "Büro", "conf": 0.9},
        {"box": [20, 32, 80, 44], "text": "45-03-012", "conf": 0.9}]}}}
    (tmp_path / "c.json").write_text(json.dumps(cache))
    texts = label_texts(cache)
    assert texts == ["büro"]
    features, layout = build_text_features(texts, ["Büro", "WC"], _FakeEncoder())
    save_features(tmp_path / "f.npz", texts, features, layout)
    provider = OcrProvider([tmp_path / "c.json"], [tmp_path / "f.npz"])
    boxes, feats, conf = provider.get("some/dir/p1.png", 200, 100)
    assert boxes.shape == (1, 4) and np.allclose(boxes[0], [0.1, 0.1, 0.3, 0.3])
    assert feats.shape == (1, provider.dim) and provider.dim == 3 + 256 + 2
    with pytest.raises(ValueError):
        provider.get("p1.png", 201, 100)


def _plan_with_label(word="Buero"):
    image = Image.new("RGB", (400, 300), (245, 200, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 350, 250], outline=(0, 0, 0), width=4)
    font_path = find_font()
    from PIL import ImageFont
    font = ImageFont.truetype(font_path, 22)
    draw.text((120, 120), word, fill=(30, 30, 30), font=font)
    x0, y0, x1, y1 = draw.textbbox((120, 120), word, font=font)
    annotation = {"id": 7, "category_id": 3, "bbox": [50, 50, 300, 200],
                  "segmentation": [[50, 50, 350, 50, 350, 250, 50, 250]]}
    token = {"box": [x0 - 3, y0 - 4, x1 + 3, y1 + 4], "text": word, "conf": 0.95}
    return np.asarray(image), annotation, token


def test_convert_room_erases_old_label_and_renders_new_name_inside_room():
    image, annotation, token = _plan_with_label()
    result = convert_room(image, annotation, [], [token], "Abstellraum", {"eraser": "telea"}, seed=1)
    assert result["status"] == "generated", result
    edited, edit_mask = result["image"], result["edit_mask"]
    room = np.zeros(image.shape[:2], bool); room[50:251, 50:351] = True
    assert not (edited != image).any(-1)[~room].any()
    x0, y0, x1, y1 = (int(v) for v in token["box"])
    patch = make_patch(edited, edit_mask)
    assert np.array_equal(apply_patch(image, patch), np.where(edit_mask[..., None], edited, image))
    assert result["render"]["text_box"][0] >= 50


def test_render_rejects_text_that_cannot_fit():
    interior = np.zeros((100, 100), bool); interior[40:50, 40:50] = True
    out, _, reason = render_word(np.full((100, 100, 3), 255, np.uint8), interior, "Versammlungsraum",
                                 [40, 40, 50, 50], (0, 0, 0), find_font())
    assert out is None and reason == "text_does_not_fit_inside_room"


def test_verify_reading_requires_new_word_and_absence_of_old():
    ok = verify_reading([{"text": "Abstellraum"}, {"text": "45-03-012"}], "Abstellraum", ["Büro"])
    assert ok["passed"]
    stale = verify_reading([{"text": "Abstellraum"}, {"text": "Buro"}], "Abstellraum", ["Büro"])
    assert not stale["passed"]
    unread = verify_reading([{"text": "Abst"}], "Abstellraum", ["Büro"])
    assert not unread["passed"]


@pytest.mark.parametrize("name", ["e18_text_relabel", "e19_text_aware", "e20_text_aware_relabel"])
def test_ocr_experiment_configs_are_whitelisted(name):
    _validate_experiment(load_yaml(resolve_path(f"configs/experiments_ocr/{name}.yaml")))


def test_text_head_is_required_exactly_for_text_arms():
    cfg = load_yaml(resolve_path("configs/experiments_ocr/e19_text_aware.yaml"))
    cfg.pop("text_head")
    with pytest.raises(ValueError):
        _validate_experiment(cfg)
    e0 = load_yaml(resolve_path("configs/experiments/e0_original.yaml"))
    e0["text_head"] = {"hidden": 8}
    with pytest.raises(ValueError):
        _validate_experiment(e0)


def test_room_label_words_merges_hyphenation_and_drops_ocr_debris():
    tok = lambda text, y, conf=0.9, x=10: {"text": text, "conf": conf, "box": [x, y, x + 50, y + 12]}
    assert room_label_words([tok("Abstell-", 0), tok("raum", 14), tok("45-03-012", 28)]) == ["Abstellraum"]
    assert room_label_words([tok("arde", 0), tok("robe", 14), tok("RS", 28), tok("[Abstell", 42)]) == []
    assert room_label_words([tok("Foyer", 0), tok("Empfang", 14, conf=0.3)]) == ["Foyer"]
    assert room_label_words([tok("Abst.", 0)]) == ["Abst"]
    assert room_label_words([tok("Abstell", 0), tok("raum", 14)]) == ["Abstellraum"]
    assert room_label_words([tok("T O-RS", 0)]) == []
