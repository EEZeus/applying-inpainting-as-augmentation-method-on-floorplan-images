"""Dependency-free OCR text normalisation, token geometry and hashed char n-grams.

Room labels in the plans are short German words ("Büro", "Spindraum H",
"Putzr.") next to room numbers ("45-03-012"). Numbers, grid letters and symbol
codes carry no room-type information, so a token is kept as a *label token*
only if at least two letters survive normalisation.
"""
import difflib
import re
import unicodedata
import zlib

import numpy as np

_DIGIT_RUN = re.compile(r"[0-9][0-9\-./:,]*")
_NON_WORD = re.compile(r"[^a-zäöüß ]+")
_SPACES = re.compile(r"\s+")
_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def normalize_text(text):
    """Lowercase, NFKC, digits removed, punctuation to spaces, umlauts kept."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _DIGIT_RUN.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def fold_umlauts(text):
    """ä->ae etc.; OCR engines often drop or replace umlauts."""
    return normalize_text(text).translate(_FOLD)


def is_label_text(text, min_letters=2):
    return sum(ch.isalpha() for ch in normalize_text(text)) >= min_letters


def strip_digits(text):
    """Raw-case text without room-number runs (used to pick render words)."""
    return _SPACES.sub(" ", _DIGIT_RUN.sub(" ", unicodedata.normalize("NFKC", str(text)))).strip(" -./:,")


def text_similarity(a, b):
    """Similarity of two OCR strings after umlaut folding (0..1)."""
    a, b = fold_umlauts(a).replace(" ", ""), fold_umlauts(b).replace(" ", "")
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def char_ngram_hash(text, dim=256, n=3):
    """L2-normalised hashed bag of character n-grams of the folded text.

    Captures abbreviations and compounds ("putzr", "spindraum") that a sentence
    encoder may embed poorly. zlib.crc32 is stable across processes.
    """
    vector = np.zeros(dim, dtype=np.float32)
    folded = fold_umlauts(text)
    if not folded:
        return vector
    for word in folded.split():
        padded = f"<{word}>"
        for i in range(max(1, len(padded) - n + 1)):
            gram = padded[i:i + n]
            vector[zlib.crc32(gram.encode("utf-8")) % dim] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def quad_to_box(points):
    """EasyOCR 4-point polygon -> axis-aligned [x0, y0, x1, y1]."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return [float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max())]


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_containment(inner, outer):
    """Fraction of ``inner`` covered by ``outer``."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


def merge_tile_tokens(tokens, iou_threshold=0.3, containment_threshold=0.7):
    """Greedy de-duplication of tokens read twice in overlapping tiles.

    Higher confidence wins; ties prefer the longer string (a token cut at a
    tile border is shorter than the same token read whole).
    """
    order = sorted(tokens, key=lambda t: (-float(t["conf"]), -len(t["text"])))
    kept = []
    for token in order:
        duplicate = False
        for other in kept:
            if (box_iou(token["box"], other["box"]) >= iou_threshold
                    or box_containment(token["box"], other["box"]) >= containment_threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(token)
    return sorted(kept, key=lambda t: (t["box"][1], t["box"][0]))


def tile_origins(length, tile, overlap):
    """Start offsets covering [0, length) with tiles of size ``tile``."""
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def token_centers(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def tokens_in_mask(boxes, mask):
    """Indices of tokens whose box centre falls on a true pixel of ``mask``."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    centers = np.rint(token_centers(boxes)).astype(np.int64)
    height, width = mask.shape
    xs = np.clip(centers[:, 0], 0, width - 1)
    ys = np.clip(centers[:, 1], 0, height - 1)
    return np.nonzero(mask[ys, xs])[0]
