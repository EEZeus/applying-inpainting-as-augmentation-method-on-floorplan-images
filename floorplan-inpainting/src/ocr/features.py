"""Per-token text features and the OCR provider used by datasets and models.

Feature of one label token (fixed a priori, not tuned on held-out plans):
  [ multilingual-e5-small embedding (384) | hashed char trigrams (256) |
    cosine similarity of the e5 embedding to each class name (n_classes) ]
The class-name similarity is a zero-shot prior; class names are label
metadata, not annotations, so using them is not leakage.
"""
import json
from pathlib import Path

import numpy as np

from .text import char_ngram_hash, is_label_text, normalize_text

E5_MODEL = "intfloat/multilingual-e5-small"
HASH_DIM = 256


class E5Encoder:
    def __init__(self, model_name=E5_MODEL, device=None, cache_dir=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device).eval()
        self.model_name = model_name

    def encode(self, texts, batch_size=64):
        torch = self.torch
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [f"query: {t}" for t in texts[start:start + batch_size]]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=32,
                                         return_tensors="pt").to(self.device)
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                outputs.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy())
        dim = self.model.config.hidden_size
        return np.concatenate(outputs) if outputs else np.zeros((0, dim), dtype=np.float32)


def build_text_features(texts, class_names, encoder):
    """Feature matrix for unique normalised texts; returns (features, layout)."""
    names = [normalize_text(name) for name in class_names]
    text_emb = encoder.encode(list(texts))
    name_emb = encoder.encode(names)
    hashed = np.stack([char_ngram_hash(t, HASH_DIM) for t in texts]) if texts else np.zeros((0, HASH_DIM))
    similarity = text_emb @ name_emb.T
    features = np.concatenate([text_emb, hashed, similarity], axis=1).astype(np.float32)
    e5_dim, n_classes = text_emb.shape[1], len(class_names)
    layout = {"e5": [0, e5_dim], "hash": [e5_dim, e5_dim + HASH_DIM],
              "class_sim": [e5_dim + HASH_DIM, e5_dim + HASH_DIM + n_classes],
              "class_names": list(class_names), "encoder": encoder.model_name}
    return features, layout


def label_texts(cache):
    """Unique normalised label strings of an OCR cache."""
    found = set()
    for entry in cache["images"].values():
        for token in entry["tokens"]:
            if is_label_text(token["text"]):
                found.add(normalize_text(token["text"]))
    return sorted(found)


def save_features(path, texts, features, layout):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, texts=np.asarray(texts, dtype=object), features=features,
             layout=np.asarray(json.dumps(layout)))


def load_features(path):
    data = np.load(path, allow_pickle=True)
    texts = [str(t) for t in data["texts"]]
    return dict(zip(texts, data["features"])), json.loads(str(data["layout"]))


class OcrProvider:
    """Label tokens of a plan as normalised boxes, features and confidences.

    Several (cache, features) pairs can be merged, e.g. original plans plus
    synthetic plans. Lookup is by file basename.
    """

    def __init__(self, cache_paths, feature_paths):
        self.images, self.lookup, self.layout = {}, {}, None
        for path in cache_paths:
            cache = json.loads(Path(path).read_text(encoding="utf-8"))
            for name, entry in cache["images"].items():
                if name in self.images and self.images[name]["tokens"] != entry["tokens"]:
                    raise ValueError(f"Conflicting OCR entries for {name}")
                self.images[name] = entry
        for path in feature_paths:
            lookup, layout = load_features(path)
            if self.layout is not None and layout != self.layout:
                raise ValueError("OCR feature files use different layouts")
            self.layout = layout
            self.lookup.update(lookup)
        if self.layout is None:
            raise ValueError("No OCR feature file given")
        self.dim = int(self.layout["class_sim"][1])

    def __contains__(self, file_name):
        return Path(file_name).name in self.images

    def get(self, file_name, width=None, height=None):
        name = Path(file_name).name
        if name not in self.images:
            raise KeyError(f"No OCR entry for {name}; run scripts/ocr_cache.py on this dataset")
        entry = self.images[name]
        if width is not None and (int(entry["width"]), int(entry["height"])) != (int(width), int(height)):
            raise ValueError(f"OCR cache size mismatch for {name}")
        boxes, feats, conf = [], [], []
        scale = np.array([entry["width"], entry["height"], entry["width"], entry["height"]], dtype=np.float32)
        for token in entry["tokens"]:
            if not is_label_text(token["text"]):
                continue
            key = normalize_text(token["text"])
            if key not in self.lookup:
                raise KeyError(f"Missing text feature for {key!r}; rerun scripts/ocr_embed.py")
            boxes.append(np.asarray(token["box"], dtype=np.float32) / scale)
            feats.append(self.lookup[key])
            conf.append(float(token["conf"]))
        if not boxes:
            return (np.zeros((0, 4), np.float32), np.zeros((0, self.dim), np.float32),
                    np.zeros(0, np.float32))
        return (np.clip(np.stack(boxes), 0.0, 1.0).astype(np.float32),
                np.stack(feats).astype(np.float32), np.asarray(conf, dtype=np.float32))

    def raw_tokens(self, file_name):
        return self.images[Path(file_name).name]["tokens"]


def provider_from_config(ocr_cfg, root_resolver):
    return OcrProvider([root_resolver(p) for p in ocr_cfg["caches"]],
                       [root_resolver(p) for p in ocr_cfg["features"]])
