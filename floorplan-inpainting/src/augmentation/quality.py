from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    rejection_reason: str | None
    metrics: dict


def _fraction(mask):
    return float(mask.mean()) if mask.size else 0.0


def evaluate_quality(original, augmented, safe_mask, boundary_mask, dark_threshold, config):
    if original.shape != augmented.shape:
        return QualityResult(False, "invalid_output_dimensions", {})
    changed = np.any(original != augmented, axis=2)
    outside = ~safe_mask
    outside_ratio = float(changed[outside].mean()) if outside.any() else 0.0
    boundary_ratio = float(changed[boundary_mask].mean()) if boundary_mask.any() else 0.0

    safe_count = int(safe_mask.sum())
    if safe_count:
        absolute = np.abs(original[safe_mask].astype(np.int16) - augmented[safe_mask].astype(np.int16))
        changed_inside = float(changed[safe_mask].mean())
        mean_absolute = float(absolute.mean())
        before_values = original[safe_mask]
        after_values = augmented[safe_mask]
        before_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)[safe_mask]
        after_gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)[safe_mask]
        near_white_before = _fraction(np.all(before_values >= 250, axis=1))
        near_white_after = _fraction(np.all(after_values >= 250, axis=1))
        dark_before = _fraction(before_gray <= dark_threshold)
        dark_after = _fraction(after_gray <= dark_threshold)
        generated_std = float(after_gray.astype(np.float32).std())
        source_dark = before_gray <= dark_threshold
        source_dark_count = int(source_dark.sum())
        source_dark_retention = (
            float((after_gray[source_dark] <= dark_threshold).mean()) if source_dark_count else 0.0
        )
        saturated_before = _fraction(np.ptp(before_values.astype(np.int16), axis=1) > 80)
        saturated_after = _fraction(np.ptp(after_values.astype(np.int16), axis=1) > 80)
    else:
        changed_inside = mean_absolute = generated_std = 0.0
        near_white_before = near_white_after = dark_before = dark_after = 0.0
        saturated_before = saturated_after = 0.0
        source_dark_count = 0
        source_dark_retention = 0.0

    metrics = {
        "outside_change_ratio": outside_ratio,
        "protected_boundary_change_ratio": boundary_ratio,
        "boundary_change_ratio": boundary_ratio,
        "changed_fraction_inside_safe_mask": changed_inside,
        "mean_absolute_change_inside_safe_mask": mean_absolute,
        "near_white_fraction_before": near_white_before,
        "near_white_fraction_after": near_white_after,
        "dark_pixel_density_before": dark_before,
        "dark_pixel_density_after": dark_after,
        "generated_region_std": generated_std,
        "changed_fraction": changed_inside,
        "mean_absolute_change": mean_absolute,
        "dark_pixel_fraction_before": dark_before,
        "dark_pixel_fraction_after": dark_after,
        "white_fraction_before": near_white_before,
        "white_fraction_after": near_white_after,
        "saturated_fraction_before": saturated_before,
        "saturated_fraction_after": saturated_after,
        "source_dark_pixel_count_inside_safe_mask": source_dark_count,
        "source_dark_retention_fraction": source_dark_retention,
    }
    checks = [
        (outside_ratio != 0.0, "outside_pixels_changed"),
        (boundary_ratio != 0.0, "protected_boundary_changed"),
        (safe_count == 0, "empty_safe_mask"),
        (
            changed_inside < float(config.get("min_changed_fraction_inside", 0.01)),
            "generated_region_almost_identical",
        ),
        (
            mean_absolute < float(config.get("min_mean_absolute_change", 1.0)),
            "generated_region_almost_identical",
        ),
        (
            source_dark_count >= int(config.get("min_source_dark_pixels_for_retention_check", 5))
            and source_dark_retention > float(config.get("max_source_dark_retention_fraction", 1.0)),
            "source_semantic_strokes_retained",
        ),
        (generated_std < float(config.get("min_generated_region_std", 2.0)), "generated_region_uniform"),
        (
            near_white_after > float(config.get("max_near_white_fraction_after", 0.995)),
            "generated_region_blank",
        ),
        (
            dark_after > config.get("max_dark_fraction_after", 0.50)
            and dark_after - dark_before > config.get("max_dark_fraction_increase", 0.30),
            "extreme_dark_filling",
        ),
        (
            saturated_after > config.get("max_saturated_fraction_after", 0.50)
            and saturated_after - saturated_before > config.get("max_saturated_fraction_increase", 0.30),
            "extreme_color_filling",
        ),
    ]
    reason = next((reason for failed, reason in checks if failed), None)
    return QualityResult(reason is None, reason, metrics)
