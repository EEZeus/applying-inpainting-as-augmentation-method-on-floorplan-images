"""COCO AP on arbitrary image multisets from one COCOeval.evaluate() pass.

``COCOeval.evaluate()`` matches detections per (image, category), so AP over any
subset or bootstrap resample of images (duplicates allowed) only needs those
per-image records re-accumulated. ``PerImageEval`` replicates ``accumulate()`` and
the AP / AP50 summary for area 'all', maxDets=100; ``check_against_cocoeval``
verifies this on the full set.
"""

import contextlib
import gzip
import io
import json

import numpy as np
from pycocotools.cocoeval import COCOeval


def load_dump(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


class PerImageEval:
    def __init__(self, coco_gt, results, image_ids, iou_type="segm"):
        self.image_ids = sorted(int(i) for i in set(image_ids))
        self.position = {image_id: index for index, image_id in enumerate(self.image_ids)}
        self.empty = not results
        self.evaluator = None
        if self.empty:
            self.category_ids = sorted(coco_gt.getCatIds())
            return
        evaluator = COCOeval(coco_gt, coco_gt.loadRes(results), iou_type)
        evaluator.params.imgIds = list(self.image_ids)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate()
        self.evaluator = evaluator
        params = evaluator.params
        self.category_ids = list(params.catIds)
        self.area_index = [index for index, label in enumerate(params.areaRngLbl) if label == "all"][0]
        self.n_area = len(params.areaRng)
        self.max_det = params.maxDets[-1]
        self.iou_thresholds = params.iouThrs
        self.rec_thresholds = params.recThrs
        self.iou50 = int(np.argmin(np.abs(params.iouThrs - 0.5)))

    def precision(self, image_ids):
        """precision[t, r, k] over the given multiset of image ids (-1 = undefined)."""
        if self.empty:
            return None
        n_images = len(self.image_ids)
        T, R, K = len(self.iou_thresholds), len(self.rec_thresholds), len(self.category_ids)
        precision = -np.ones((T, R, K))
        positions = [self.position[int(i)] for i in image_ids]
        for k in range(K):
            base = k * self.n_area * n_images + self.area_index * n_images
            entries = [self.evaluator.evalImgs[base + i] for i in positions]
            entries = [e for e in entries if e is not None]
            if not entries:
                continue
            scores = np.concatenate([e["dtScores"][0 : self.max_det] for e in entries])
            order = np.argsort(-scores, kind="mergesort")
            dtm = np.concatenate([e["dtMatches"][:, 0 : self.max_det] for e in entries], axis=1)[:, order]
            dtig = np.concatenate([e["dtIgnore"][:, 0 : self.max_det] for e in entries], axis=1)[:, order]
            gtig = np.concatenate([e["gtIgnore"] for e in entries])
            npig = np.count_nonzero(gtig == 0)
            if npig == 0:
                continue
            tps = np.logical_and(dtm, np.logical_not(dtig))
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtig))
            tp_sum = np.cumsum(tps, axis=1).astype(float)
            fp_sum = np.cumsum(fps, axis=1).astype(float)
            for t in range(T):
                tp, fp = tp_sum[t], fp_sum[t]
                nd = len(tp)
                q = np.zeros(R)
                if nd:
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    pr = np.maximum.accumulate(pr[::-1])[::-1]
                    inds = np.searchsorted(rc, self.rec_thresholds, side="left")
                    valid = inds < nd
                    q[valid] = pr[inds[valid]]
                precision[t, :, k] = q
        return precision

    def ap(self, image_ids):
        precision = self.precision(image_ids)
        if precision is None:
            return {"AP": 0.0, "AP50": 0.0}

        def mean_valid(values):
            values = values[values > -1]
            return float(values.mean()) if values.size else -1.0

        return {"AP": mean_valid(precision), "AP50": mean_valid(precision[self.iou50])}

    def check_against_cocoeval(self, tolerance=1e-9):
        """Assert that the replica equals COCOeval on the full image set."""
        if self.empty:
            return True
        with contextlib.redirect_stdout(io.StringIO()):
            self.evaluator.accumulate()
            self.evaluator.summarize()
        mine = self.ap(self.image_ids)
        reference = (float(self.evaluator.stats[0]), float(self.evaluator.stats[1]))
        if abs(mine["AP"] - reference[0]) > tolerance or abs(mine["AP50"] - reference[1]) > tolerance:
            raise AssertionError(f"subset AP replica {mine} != COCOeval {reference}")
        return True
