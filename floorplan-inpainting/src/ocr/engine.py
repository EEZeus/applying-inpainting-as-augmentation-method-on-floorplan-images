"""Tiled full-resolution OCR with EasyOCR (German + English).

Plans are read at native resolution in overlapping tiles because the detector
input (max side 1600 px) shrinks room labels to a few pixels. OCR sees pixels
only, never annotations.
"""

import numpy as np

from .text import merge_tile_tokens, quad_to_box, tile_origins


class EasyOcrEngine:
    def __init__(
        self,
        languages=("de", "en"),
        gpu=True,
        model_dir=None,
        allow_download=False,
        tile=1600,
        overlap=256,
        rotation_info=(90, 270),
        min_conf=0.10,
        batch_size=8,
    ):
        import easyocr  # imported lazily: only the `ocr` environment has it

        self.reader = easyocr.Reader(
            list(languages),
            gpu=gpu,
            model_storage_directory=model_dir,
            download_enabled=allow_download,
            verbose=False,
        )
        self.tile, self.overlap = int(tile), int(overlap)
        self.rotation_info = list(rotation_info) if rotation_info else None
        self.min_conf = float(min_conf)
        self.batch_size = int(batch_size)
        self.metadata = {
            "engine": "easyocr",
            "version": getattr(easyocr, "__version__", "unknown"),
            "languages": list(languages),
            "tile": self.tile,
            "overlap": self.overlap,
            "rotation_info": self.rotation_info,
            "min_conf": self.min_conf,
        }

    def read_array(self, image_rgb):
        """OCR one RGB uint8 array; boxes are in its pixel coordinates."""
        height, width = image_rgb.shape[:2]
        tokens = []
        for y0 in tile_origins(height, self.tile, self.overlap):
            for x0 in tile_origins(width, self.tile, self.overlap):
                crop = np.ascontiguousarray(image_rgb[y0 : y0 + self.tile, x0 : x0 + self.tile])
                results = self.reader.readtext(
                    crop,
                    detail=1,
                    paragraph=False,
                    rotation_info=self.rotation_info,
                    batch_size=self.batch_size,
                )
                for points, text, conf in results:
                    if float(conf) < self.min_conf or not str(text).strip():
                        continue
                    x_min, y_min, x_max, y_max = quad_to_box(points)
                    tokens.append(
                        {
                            "box": [x_min + x0, y_min + y0, x_max + x0, y_max + y0],
                            "text": str(text),
                            "conf": float(conf),
                        }
                    )
        return merge_tile_tokens(tokens)
