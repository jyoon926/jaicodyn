#!/usr/bin/env python3

import numpy as np

from jaicodyn.data import available
from jaicodyn.data.labels import target_labels
from jaicodyn.decode import Params, decode
from jaicodyn.features import build


def evaluate_decoder_comparison(anchor_closer_m=3.0):
    rows = []
    p = Params(anchor_closer_m=anchor_closer_m)
    for match in available():
        if not match.source.startswith("idsse"):
            continue
        feats = build(match)
        gt = target_labels(match, "strict")
        original = np.asarray(decode(feats, match.fps, match.pitch, p=p), dtype=object)
        with_anchor = np.asarray(decode(feats, match.fps, match.pitch, p=p, anchor_rules=True), dtype=object)
        rows.append({
            "match": match.key,
            "original": float((original == gt).mean()),
            "with_anchor": float((with_anchor == gt).mean()),
        })

    original_scores = np.asarray([r["original"] for r in rows], dtype=float)
    anchor_scores = np.asarray([r["with_anchor"] for r in rows], dtype=float)

    print("match\toriginal_viterbi\tviterbi_plus_anchor")
    for row in rows:
        print(f"{row['match']}\t{row['original']:.6f}\t{row['with_anchor']:.6f}")

    print("\nSUMMARY")
    print(f"average_original={float(original_scores.mean()):.6f}")
    print(f"average_with_anchor={float(anchor_scores.mean()):.6f}")
    print(f"delta={float(anchor_scores.mean() - original_scores.mean()):.6f}")
    return rows


if __name__ == "__main__":
    evaluate_decoder_comparison()
