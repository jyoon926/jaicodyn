"""Hardcoded anchor rules for possession assignment.

The anchor model is intentionally narrow: it marks only the frames where one side is so clearly
nearest and sustained that we are comfortable locking the possession in. All other frames remain
unresolved and are filled by the Viterbi decoder in between those anchor points.

This intentionally ignores dead-ball logic for now.
"""

from __future__ import annotations

import numpy as np


def evaluate_anchor_rules(source_prefix="idsse", return_rows=False):
    """Evaluate the anchor rules as an abstaining classifier.

    Returns the per-match accuracy on frames where the anchor rule makes a decision, plus the
    fraction of all frames it chose to label. This is the correct metric because the anchor layer
    intentionally abstains when evidence is not strong enough.
    """
    from jaicodyn.data import available
    from jaicodyn.data.labels import target_labels
    from jaicodyn.features import build

    rows = []
    for match in available():
        if not match.source.startswith(source_prefix):
            continue
        feats = build(match)
        anchors = anchor_labels(feats, match.fps, pitch=match.pitch)
        gt = target_labels(match, "strict")
        pred = np.asarray(anchors, dtype=object)
        mask = pred != None
        if not mask.any():
            rows.append({
                "match": match.key,
                "accuracy": None,
                "predicted_frames": 0,
                "total_frames": len(gt),
                "coverage": 0.0,
            })
            continue
        correct = int(np.sum(pred[mask] == gt[mask]))
        accuracy = correct / mask.sum()
        rows.append({
            "match": match.key,
            "accuracy": float(accuracy),
            "predicted_frames": int(mask.sum()),
            "total_frames": len(gt),
            "coverage": float(mask.mean()),
        })

    accs = np.asarray([r["accuracy"] for r in rows if r["accuracy"] is not None], dtype=float)
    coverages = np.asarray([r["coverage"] for r in rows], dtype=float)
    summary = {
        "match_count": len(rows),
        "average_accuracy": float(np.mean(accs)) if accs.size else None,
        "min_accuracy": float(np.min(accs)) if accs.size else None,
        "max_accuracy": float(np.max(accs)) if accs.size else None,
        "mean_coverage": float(np.mean(coverages)) if coverages.size else None,
        "prediction_percent": float(np.mean(coverages) * 100.0) if coverages.size else None,
    }

    if return_rows:
        return rows, summary

    print("match\taccuracy\tpredicted_frames\ttotal_frames\tcoverage\tprediction_percent")
    for row in rows:
        pct = row["coverage"] * 100.0
        print(f"{row['match']}\t{row['accuracy'] if row['accuracy'] is not None else 'NA'}\t"
              f"{row['predicted_frames']}\t{row['total_frames']}\t{row['coverage']:.6f}\t{pct:.2f}%")
    print("\nSUMMARY")
    print(f"match_count={summary['match_count']}")
    print(f"average_accuracy={summary['average_accuracy']}")
    print(f"min_accuracy={summary['min_accuracy']}")
    print(f"max_accuracy={summary['max_accuracy']}")
    print(f"mean_coverage={summary['mean_coverage']}")
    print(f"prediction_percent={summary['prediction_percent']}")
    return rows, summary


def _ball_speed(feats, fps):
    """Ball speed in m/s, using a previous-frame delta for each row."""
    xy = feats[["ball_x", "ball_y"]].to_numpy(float)
    out = np.zeros(len(feats), dtype=float)
    valid = ~np.isnan(xy[:, 0]) & ~np.isnan(xy[:, 1])
    if not valid.any():
        return out
    for i in range(1, len(feats)):
        if valid[i - 1] and valid[i]:
            out[i] = np.linalg.norm(xy[i] - xy[i - 1]) * fps
    return out


def _ball_in_play(feats, fps, pitch=None):
    """True only on live-ball frames. If pitch is known, also exclude dead-ball cues."""
    speed = _ball_speed(feats, fps)
    live = speed > 0.5
    if pitch is not None:
        from .dead import cues
        live &= cues(feats, fps, pitch)["dead"].to_numpy(float) <= 0.0
    return live


def _periods(feats):
    period = feats["period"].to_numpy()
    cuts = np.flatnonzero(period[1:] != period[:-1]) + 1
    return zip(np.r_[0, cuts], np.r_[cuts, len(feats)])


def _min_anchor_control(fps, p):
    """Minimum sustained control needed before we lock a possession anchor.

    The default anchor layer is intentionally conservative: a short burst of control is not enough
    to overrule the Viterbi decoder. We require several turnover windows of clear control before
    pinning the state, leaving short touches and marginal cases unresolved.
    """
    min_control_s = max(1.5, 3.0 * p.turnover_s)
    return max(2, int(round(min_control_s * fps)))


def anchor_labels(feats, fps, p=None, pitch=None):
    """Return H/A/None on each row, with fixed labels only when the anchor rule is strong.

    The strong-rule gate is intentionally conservative: only live-ball frames with a clear margin and
    a ball speed roughly matching the visible player speed are eligible.
    """
    if p is None:
        from . import Params
        p = Params()

    live = _ball_in_play(feats, fps, pitch)
    out = np.full(len(feats), None, dtype=object)
    required = _min_anchor_control(fps, p)

    for start, stop in _periods(feats):
        rows = feats.iloc[start:stop].reset_index(drop=True)
        near_h = rows["near_H"].to_numpy(float)
        near_a = rows["near_A"].to_numpy(float)
        run_team = None
        run_start = None

        for i in range(len(rows)):
            idx = start + i
            h = near_h[i]
            a = near_a[i]

            if np.isnan(h) or np.isnan(a):
                if run_team is not None and (i - run_start) >= required:
                    out[start + run_start:start + i] = run_team
                run_team = None
                run_start = None
                continue

            if not live[idx]:
                if run_team is not None and (i - run_start) >= required:
                    out[start + run_start:start + i] = run_team
                run_team = None
                run_start = None
                continue

            if h <= p.gate_near_m and a > h + p.anchor_closer_m:
                candidate = "H"
            elif a <= p.gate_near_m and h > a + p.anchor_closer_m:
                candidate = "A"
            else:
                candidate = None

            if candidate is None:
                if run_team is not None and (i - run_start) >= required:
                    out[start + run_start:start + i] = run_team
                run_team = None
                run_start = None
                continue

            if run_team is None:
                run_team = candidate
                run_start = i
            elif candidate != run_team:
                if (i - run_start) >= required:
                    out[start + run_start:start + i] = run_team
                run_team = candidate
                run_start = i

        if run_team is not None and (len(rows) - run_start) >= required:
            out[start + run_start:stop] = run_team

    return out


def decode_anchor_rules(feats, fps, pitch=None, tags=None, p=None, gap_probs=None,
                       restart_p_home=None, team_evidence=None):
    """Resolve full H/A/D labels with anchor points, leaving all other frames to Viterbi.

    The hardcoded rules only pin some frames as fixed labels and leave the rest unresolved. The
    Viterbi decoder then runs only on the gaps between those fixed anchors, using the anchor
    states as boundary conditions.
    """
    if p is None:
        from . import Params
        p = Params()

    from . import H, A, STATES, emissions, switch_costs, viterbi, _periods

    anchors = anchor_labels(feats, fps, p=p, pitch=pitch)
    labels = np.full(len(feats), "D", dtype=object)

    for start, stop in _periods(feats):
        period = feats.iloc[start:stop].reset_index(drop=True)
        arr = anchors[start:stop]
        fixed = np.flatnonzero(np.isin(arr, ["H", "A"]))
        if fixed.size == 0:
            score = emissions(period, fps, pitch, tags, p,
                              gap_probs[start:stop] if gap_probs is not None else None,
                              restart_p_home[start:stop] if restart_p_home is not None else None,
                              team_evidence[start:stop] if team_evidence is not None else None)
            labels[start:stop] = STATES[viterbi(score, switch_costs(p))]
            continue

        resolved = np.full(len(period), None, dtype=object)
        for idx in fixed:
            resolved[idx] = arr[idx]

        anchor_positions = fixed.tolist()
        for left_idx, right_idx in zip([-1] + anchor_positions, anchor_positions + [len(period)]):
            gap_start = left_idx + 1 if left_idx >= 0 else 0
            gap_stop = right_idx
            if gap_start >= gap_stop:
                continue
            slice_ = period.iloc[gap_start:gap_stop]
            score = emissions(slice_, fps, pitch, tags, p,
                              gap_probs[start + gap_start:start + gap_stop] if gap_probs is not None else None,
                              restart_p_home[start + gap_start:start + gap_stop] if restart_p_home is not None else None,
                              team_evidence[start + gap_start:start + gap_stop] if team_evidence is not None else None)

            if left_idx >= 0:
                left_state = H if arr[left_idx] == "H" else A
                score[0, np.arange(3) != left_state] = -np.inf
            if right_idx < len(period):
                right_state = H if arr[right_idx] == "H" else A
                score[-1, np.arange(3) != right_state] = -np.inf

            segment = viterbi(score, switch_costs(p))
            resolved[gap_start:gap_stop] = STATES[segment]

        labels[start:stop] = resolved

    return labels
