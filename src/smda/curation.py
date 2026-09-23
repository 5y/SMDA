"""Explicit quantitative criteria and strict spreadsheet boolean parsing."""
import numpy as np


def strict_bool(value):
    value = str(value).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"Expected TRUE/FALSE, got {value!r}.")
    return value == "true"


def quantitative_mask(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid influence scores.")
    h, b = labels == "harmful_refusal", labels == "harmless_compliance"
    if not h.any() or not b.any() or not (h | b).all():
        raise ValueError("Both recognized SFT classes required.")
    mh, mb = float(np.median(scores[h])), float(np.median(scores[b]))
    return (h & (scores < mb)) | (b & (scores > mh)), {"harmful_median": mh, "harmless_median": mb}


def corrected_csv_mask(rows):
    """Use J OR L, never the corrupted legacy misalligned_quant column."""
    result = []
    for r in rows:
        j = strict_bool(r["more_negative_than_harmless_median_for_harmful"])
        l = strict_bool(r["more_positive_than_harmful_median_for_harmless"])
        if (j and r["label"] != "harmful_refusal") or (l and r["label"] != "harmless_compliance"):
            raise ValueError("Removal flag contradicts pair label.")
        result.append(j or l)
    return np.array(result, dtype=bool)
