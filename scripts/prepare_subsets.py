"""Prepare full / corrected quantitative / primary-rater qualitative manifests."""
import argparse
from collections import Counter
from pathlib import Path
from smda.data import read_rows, validate_pairs, write_json
from smda.curation import corrected_csv_mask


def run(source, output):
    # Discard spreadsheet summary rows, but never silently select the first 200.
    rows = [r for r in read_rows(source) if str(r.get("prompt") or "").strip() and str(r.get("response") or "").strip()]
    rows = validate_pairs(rows)
    if len(rows) != 200 or Counter(r["label"] for r in rows) != {"harmful_refusal":100, "harmless_compliance":100}:
        raise ValueError("Expected exactly the paper's 100+100 SFT pairs.")
    mask = corrected_csv_mask(rows)
    removed = Counter(r["label"] for r, flag in zip(rows, mask) if flag)
    if removed != {"harmful_refusal":47, "harmless_compliance":39}:
        raise ValueError(f"Corrected quantitative counts differ from the paper: {removed}.")
    # Paper uses primary rater decisions; do not replace them with rater intersection.
    qual = []
    for r in rows:
        value = r.get("Darian_Misalligned_qual")
        v = "" if value is None else str(value).strip()
        if v not in {"", "0", "1"}:
            raise ValueError("Unexpected primary-rater annotation.")
        if v == "1" and r["label"] != "harmful_refusal":
            raise ValueError("Qualitative removals must be harmful refusal pairs.")
        qual.append(v == "1")
    if sum(qual) != 34:
        raise ValueError("Expected 34 primary-rater qualitative removals.")
    subsets = {"full": rows, "quant_removed_corrected": [r for r, flag in zip(rows, mask) if not flag],
               "qual_removed": [r for r, flag in zip(rows, qual) if not flag]}
    for name, subset in subsets.items():
        write_json(Path(output) / f"{name}.json", subset)
    write_json(Path(output) / "selection_manifest.json", {k:[r["pair_id"] for r in v] for k,v in subsets.items()})
    return {k: len(v) for k,v in subsets.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", "--csv", dest="data", required=True)
    p.add_argument("--output", default="data/curated")
    a = p.parse_args()
    print(run(a.data, a.output))
