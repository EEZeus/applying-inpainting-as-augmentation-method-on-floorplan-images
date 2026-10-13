"""Label-consistent common-to-rare synthesis guided by OCR.

The earlier strategies changed room pixels but never the printed room name,
which is the dominant class cue in these plans. Here one converted room is:

1. a geometry-compatible frequent room (classes 1-10, Strategy B screening);
2. its OCR label tokens (e.g. "Büro") are erased by diffusion inpainting
   restricted to the token boxes inside the eroded room (OpenCV Telea is the
   pre-specified fallback when the diffusion fill fails the erase QC);
3. a rare-class name sampled from the OCR vocabulary of *training* rooms of
   the target class is rendered at the measured label height and colour;
4. the annotation's category becomes the target class; geometry and every
   other annotation are unchanged.

Room numbers (digit-only tokens) are left untouched. A separate OCR pass
(scripts/text_relabel_finish.py) admits a room only if OCR reads the new name
and no longer reads the old one.
"""
import re
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.augmentation.crops import CropTransform, composite_masked
from src.augmentation.masks import rasterize_annotation, relative_erosion_pixels
from src.ocr.text import is_label_text, normalize_text, strip_digits, tokens_in_mask


def find_font(path=None):
    if path:
        return path
    from matplotlib import font_manager
    return font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"), fallback_to_default=True)


_RENDERABLE = re.compile(r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.\- ]*$")


def room_label_words(tokens, min_conf=0.6, min_letters=3):
    """Clean room names from the OCR tokens inside one room.

    Labels wrap over lines ("Abstell-" / "raum"): a token ending in a hyphen,
    or followed by a token starting in lower case, is joined with that token. A word is kept only if every
    part was read with confidence >= min_conf, it starts with a capital letter
    (German room names are nouns), has >= min_letters letters and contains no
    OCR debris characters.
    """
    labels = sorted((t for t in tokens if is_label_text(t["text"])),
                    key=lambda t: ((t["box"][1] + t["box"][3]) / 2, t["box"][0]))
    words, skip = [], set()
    for i, token in enumerate(labels):
        if i in skip:
            continue
        text, conf = strip_digits(token["text"]), float(token["conf"])
        following = labels[i + 1] if i + 1 < len(labels) else None
        continuation = following is not None and strip_digits(following["text"])[:1].islower()
        if following is not None and (str(token["text"]).rstrip().endswith("-") or continuation):
            text = text + strip_digits(following["text"]).lower()
            conf = min(conf, float(following["conf"]))
            skip.add(i + 1)
        text = " ".join(text.split()).strip(" .-")
        parts = re.split(r"[ \-]+", text)
        if (conf >= min_conf and sum(ch.isalpha() for ch in text) >= min_letters
                and _RENDERABLE.match(text) and all(sum(ch.isalpha() for ch in part) >= 2 for part in parts)):
            words.append(text)
    return words


def rare_vocabulary(coco, train_ids, ocr_tokens_by_image, target_ids, file_names, min_conf=0.6):
    """Room names printed inside real training rooms of each target class.

    Returns {category_id: Counter(word)}; falls back to the category name for
    a class whose training rooms carry no clean readable label.
    """
    names = {int(c["id"]): c["name"] for c in coco["categories"]}
    vocab = {int(c): Counter() for c in target_ids}
    sizes = {int(i["id"]): (int(i["height"]), int(i["width"])) for i in coco["images"]}
    for annotation in coco["annotations"]:
        category, image_id = int(annotation["category_id"]), int(annotation["image_id"])
        if category not in vocab or image_id not in train_ids:
            continue
        tokens = ocr_tokens_by_image.get(file_names[image_id], [])
        if not tokens:
            continue
        mask = rasterize_annotation(annotation, *sizes[image_id])
        inside = tokens_in_mask([t["box"] for t in tokens], mask)
        for word in room_label_words([tokens[i] for i in inside], min_conf):
            vocab[category][word] += 1
    for category, counter in vocab.items():
        if not counter:
            counter[names[category]] += 1
    return vocab


def sample_word(counter, rng):
    words = sorted(counter)
    weights = np.asarray([counter[w] for w in words], dtype=np.float64)
    return words[int(rng.choice(len(words), p=weights / weights.sum()))]


def label_tokens_in_room(tokens, room_mask):
    inside = tokens_in_mask([t["box"] for t in tokens], room_mask) if tokens else []
    return [tokens[i] for i in inside if is_label_text(tokens[i]["text"])]


def erase_mask_for(tokens, room_interior, height, width, pad_fraction=0.20, min_pad=2):
    mask = np.zeros((height, width), dtype=bool)
    for token in tokens:
        x0, y0, x1, y1 = token["box"]
        pad = max(min_pad, int(round(pad_fraction * (y1 - y0))))
        xa, ya = max(0, int(np.floor(x0)) - pad), max(0, int(np.floor(y0)) - pad)
        xb, yb = min(width, int(np.ceil(x1)) + pad), min(height, int(np.ceil(y1)) + pad)
        mask[ya:yb, xa:xb] = True
    return mask & room_interior


def long_line_mask(image_rgb, room, bbox, fraction=0.20, dilation=1):
    """Dark wall/door strokes (long horizontal or vertical runs) near and inside the room.

    Label glyphs are short, so they survive the morphological opening only as
    long runs of walls; erasing may touch glyphs in the erosion band next to a
    wall without touching the wall itself.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    values = gray[room]
    if values.size == 0:
        return np.zeros_like(room)
    threshold, _ = cv2.threshold(values.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (gray <= threshold).astype(np.uint8)
    width, height = (max(1, int(round(v))) for v in bbox[2:])
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, max(5, int(fraction * width))), np.uint8))
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((max(5, int(fraction * height)), 1), np.uint8))
    lines = (horizontal | vertical).astype(np.uint8)
    if dilation:
        lines = cv2.dilate(lines, np.ones((2 * dilation + 1,) * 2, np.uint8))
    return lines.astype(bool)


def ink_height(image_rgb, token):
    """Height in pixels of the dark glyph rows inside an OCR box (None if unclear)."""
    x0, y0, x1, y1 = (int(round(v)) for v in token["box"])
    patch = image_rgb[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if patch.size == 0:
        return None
    gray = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2GRAY)
    threshold, _ = cv2.threshold(gray.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rows = np.nonzero(((gray <= threshold).sum(1)) >= 1)[0]
    return float(rows[-1] - rows[0] + 1) if rows.size >= 3 else None


def font_size_matching(font_path, text, target_ink_height):
    """Font size at which ``text`` renders with the given ink height."""
    best = None
    for size in range(6, 200):
        x0, y0, x1, y1 = ImageFont.truetype(font_path, size).getbbox(text)
        if y1 - y0 > target_ink_height:
            break
        best = size
    return best or 6


def background_color(image_rgb, room_interior, exclude):
    """Median colour of light room pixels outside the label boxes."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    region = room_interior & ~exclude
    values = gray[region]
    if values.size < 20:
        return np.array([255, 255, 255], dtype=np.float64)
    light = region & (gray >= np.percentile(values, 50))
    return np.median(image_rgb[light], axis=0).astype(np.float64)


def text_color(image_rgb, token):
    x0, y0, x1, y1 = (int(round(v)) for v in token["box"])
    patch = image_rgb[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if patch.size == 0:
        return (40, 40, 40)
    gray = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2GRAY)
    threshold, _ = cv2.threshold(gray.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray <= threshold
    if dark.sum() < 5:
        return (40, 40, 40)
    # Glyph cores, not anti-aliased edges: the darkest quarter of the ink pixels.
    core = dark & (gray <= np.percentile(gray[dark], 25))
    return tuple(int(v) for v in np.median(patch[core if core.sum() >= 3 else dark], axis=0))


def erase_quality(original, erased, erase_mask, background, config):
    """Pass if the erased region is a clean, background-coloured fill."""
    if not erase_mask.any():
        return {"passed": True, "erase_pixels": 0}
    gray = cv2.cvtColor(erased, cv2.COLOR_RGB2GRAY)[erase_mask].astype(np.float64)
    region = erased[erase_mask].astype(np.float64)
    dark_fraction = float((gray < config.get("dark_level", 110)).mean())
    color_error = float(np.abs(region - background[None]).mean())
    std = float(gray.std())
    passed = (dark_fraction <= config.get("max_dark_fraction", 0.03)
              and color_error <= config.get("max_color_error", 25.0)
              and std <= config.get("max_std", 30.0))
    return {"passed": bool(passed), "erase_pixels": int(erase_mask.sum()), "dark_fraction": dark_fraction,
            "color_error": color_error, "std": std,
            "changed_fraction": float((np.abs(erased.astype(int) - original.astype(int)).sum(-1) > 0)[erase_mask].mean())}


def erase_with_diffusion(image_rgb, erase_mask, backend, prompt, seed, output_size=512, context_fraction=1.0,
                         model_mask_dilation=3):
    ys, xs = np.nonzero(erase_mask)
    bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
    # Small label boxes get at least ~96 px of surrounding context.
    side = max(bbox[2], bbox[3], 96.0)
    transform = CropTransform.from_bbox([bbox[0] - (side - bbox[2]) / 2, bbox[1] - (side - bbox[3]) / 2, side, side],
                                        image_rgb.shape[1], image_rgb.shape[0], context_fraction, output_size)
    model_image = transform.forward_image(image_rgb)
    model_mask = transform.forward_mask(erase_mask)
    if model_mask_dilation:
        kernel = np.ones((2 * model_mask_dilation + 1,) * 2, np.uint8)
        model_mask = cv2.dilate(model_mask.astype(np.uint8), kernel).astype(bool)
    generated = backend.generate(Image.fromarray(model_image), Image.fromarray(model_mask.astype(np.uint8) * 255),
                                 prompt, seed)
    restored = transform.inverse_image(np.asarray(generated))
    return composite_masked(image_rgb, restored, erase_mask, transform)


def erase_with_telea(image_rgb, erase_mask, radius=5):
    output = cv2.inpaint(image_rgb, erase_mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    result = image_rgb.copy()
    result[erase_mask] = output[erase_mask]
    return result


def render_word(image_rgb, room_interior, word, anchor_box, color, font_path, font_size=None,
                 scales=(1.0, 0.85, 0.7), min_inside=0.98):
    """Draw ``word`` near ``anchor_box`` (left-aligned, then centred); fit inside the room.

    ``font_size`` defaults to the size whose glyphs fill ~75% of the anchor box
    height. Returns (image, text_mask, info) or (None, None, reason).
    """
    height, width = room_interior.shape
    x0, y0, x1, y1 = anchor_box
    if font_size is None:
        font_size = font_size_matching(font_path, "Hg", max(6.0, 0.75 * (y1 - y0)))
    for scale in scales:
        font = ImageFont.truetype(font_path, max(6, int(round(font_size * scale))))
        bx0, by0, bx1, by1 = font.getbbox(word)
        text_w, text_h, off_x, off_y = bx1 - bx0, by1 - by0, bx0, by0
        cy = (y0 + y1) / 2
        for left in (x0, (x0 + x1) / 2 - text_w / 2):
            top = cy - text_h / 2
            rx0, ry0 = int(round(left)), int(round(top))
            rx1, ry1 = rx0 + int(text_w), ry0 + int(text_h)
            if rx0 < 0 or ry0 < 0 or rx1 > width or ry1 > height or rx1 <= rx0 or ry1 <= ry0:
                continue
            if room_interior[ry0:ry1, rx0:rx1].mean() < min_inside:
                continue
            layer = Image.new("L", (width, height), 0)
            ImageDraw.Draw(layer).text((rx0 - off_x, ry0 - off_y), word, fill=255, font=font)
            alpha = np.asarray(layer, dtype=np.float64)[..., None] / 255.0
            alpha *= room_interior[..., None]
            out = image_rgb.astype(np.float64) * (1 - alpha) + np.asarray(color, np.float64)[None, None] * alpha
            info = {"font_size": int(font.size), "text_box": [rx0, ry0, rx1, ry1], "scale": scale}
            return np.clip(np.rint(out), 0, 255).astype(np.uint8), alpha[..., 0] > 0, info
    return None, None, "text_does_not_fit_inside_room"


def convert_room(image_rgb, target_annotation, other_annotations, tokens, word, config, seed,
                 backend=None, font_path=None):
    """Edit one room. Returns dict with status, edited image (or None) and metrics."""
    height, width = image_rgb.shape[:2]
    room = rasterize_annotation(target_annotation, height, width)
    erosion = relative_erosion_pixels(target_annotation["bbox"], config.get("erosion_fraction", 0.02),
                                      config.get("min_erosion_px", 2), config.get("max_erosion_px", 12))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
    interior = cv2.erode(room.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    for other in other_annotations:
        interior &= ~rasterize_annotation(other, height, width)
    old = label_tokens_in_room(tokens, room)
    horizontal = [t for t in old if (t["box"][2] - t["box"][0]) >= 0.8 * (t["box"][3] - t["box"][1])]
    record = {"old_texts": [t["text"] for t in old], "word": word, "erosion_px": int(erosion)}
    if old and not horizontal:
        return {**record, "status": "rejected", "reason": "only_vertical_label"}
    font_path = find_font(font_path)
    font_size = None
    if old:
        anchor = max(horizontal, key=lambda t: (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]))
        anchor_box, color = anchor["box"], text_color(image_rgb, anchor)
        ink = ink_height(image_rgb, anchor)
        if ink:
            # Same font size as the erased label: its own string rendered at that size has its ink height.
            font_size = font_size_matching(font_path, strip_digits(anchor["text"]) or anchor["text"], ink)
    else:
        # Unlabelled room: place the name at the room centre at the plan's median label height.
        heights = [t["box"][3] - t["box"][1] for t in tokens if is_label_text(t["text"])]
        line = float(np.median(heights)) if heights else 20.0
        ys, xs = np.nonzero(interior)
        if xs.size == 0:
            return {**record, "status": "rejected", "reason": "empty_interior"}
        cx, cy = float(np.median(xs)), float(np.median(ys))
        anchor_box, color = [cx - 2 * line, cy - line / 2, cx + 2 * line, cy + line / 2], (40, 40, 40)
    # Glyphs may sit in the erosion band next to a wall: erase inside the room
    # minus long wall strokes and other rooms, not only inside the eroded interior.
    erasable = room & ~long_line_mask(image_rgb, room, target_annotation["bbox"])
    for other in other_annotations:
        erasable &= ~rasterize_annotation(other, height, width)
    erase = erase_mask_for(old, erasable, height, width, config.get("erase_pad_fraction", 0.20))
    background = background_color(image_rgb, interior, erase)
    erased, eraser = image_rgb, "none"
    quality = {"passed": True, "erase_pixels": 0}
    if erase.any():
        eraser = config.get("eraser", "diffusion")
        if eraser == "diffusion":
            if backend is None:
                raise ValueError("diffusion eraser needs a backend")
            erased = erase_with_diffusion(image_rgb, erase, backend, config["erase_prompt"], seed,
                                          config.get("output_size", 512))
            quality = erase_quality(image_rgb, erased, erase, background, config.get("erase_qc", {}))
            if not quality["passed"]:
                record["diffusion_erase_qc"] = quality
                eraser = "telea_fallback"
        if eraser in {"telea", "telea_fallback"}:
            erased = erase_with_telea(image_rgb, erase)
            quality = erase_quality(image_rgb, erased, erase, background, config.get("erase_qc", {}))
        if not quality["passed"]:
            return {**record, "status": "rejected", "reason": "erase_qc_failed", "eraser": eraser,
                    "erase_qc": quality}
    rendered, text_mask, info = render_word(erased, interior, word, anchor_box, color, font_path, font_size)
    if rendered is None:
        return {**record, "status": "rejected", "reason": info, "eraser": eraser, "erase_qc": quality}
    edit_mask = erase | text_mask
    outside = (rendered != image_rgb).any(-1) & ~room
    if outside.any():
        raise AssertionError("Edit changed pixels outside the target room")
    return {**record, "status": "generated", "eraser": eraser, "erase_qc": quality, "render": info,
            "text_color": list(color), "edit_mask": edit_mask, "image": rendered,
            "anchor_box": [float(v) for v in anchor_box]}


def apply_patch(image_rgb, patch):
    """Paste a stored room edit {x0, y0, pixels, mask} onto a full plan."""
    output = image_rgb.copy()
    y0, x0 = int(patch["y0"]), int(patch["x0"])
    pixels, mask = patch["pixels"], patch["mask"].astype(bool)
    region = output[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
    region[mask] = pixels[mask]
    return output


def make_patch(edited, edit_mask):
    ys, xs = np.nonzero(edit_mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return {"y0": int(y0), "x0": int(x0), "pixels": edited[y0:y1, x0:x1].copy(), "mask": edit_mask[y0:y1, x0:x1].copy()}


def verify_reading(tokens_in_room, word, old_texts, threshold=0.75):
    """OCR admission rule: the new name is read and none of the old names is."""
    from src.ocr.text import text_similarity

    texts = [t["text"] for t in tokens_in_room if is_label_text(t["text"])]
    joined = " ".join(texts)
    new_score = max([text_similarity(t, word) for t in texts] + [text_similarity(joined, word)] + [0.0])
    old_scores = [text_similarity(t, o) for t in texts for o in old_texts
                  if normalize_text(o) and text_similarity(o, word) < threshold]
    old_score = max(old_scores + [0.0])
    return {"new_word_similarity": float(new_score), "old_word_similarity": float(old_score),
            "read_texts": texts, "passed": bool(new_score >= threshold and old_score < threshold)}
