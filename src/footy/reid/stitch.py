"""Merge fragmented tracks by appearance.

Moving cameras fragment ByteTrack badly (a pan drops IoU below the match gate), so
one player becomes many short tracks. This joins non-overlapping fragments whose
appearance embeddings agree.

Deliberately conservative: a false merge fuses two players and corrupts per-player
metrics downstream, which is worse than fragmentation - the same "nullable beats
guessed" reasoning as everywhere else. A merge needs both a high similarity and a
clear margin over the runner-up.
"""

from __future__ import annotations

import numpy as np


def _find(parent: dict[int, int], t: int) -> int:
    while parent[t] != t:
        parent[t] = parent[parent[t]]
        t = parent[t]
    return t


def stitch_tracks(
    tracks: dict[int, dict],
    sim_threshold: float = 0.65,
    margin: float = 0.08,
    max_gap_frames: int = 250,
) -> dict[int, int]:
    """tid -> {'start', 'end', 'embedding'} => mapping tid -> canonical tid.

    Only merges fragments separated in time (no overlap), close enough in time,
    and whose embedding similarity clears both the absolute threshold and the
    margin over the second-best candidate for either side.
    """
    tids = sorted(tracks)
    parent = {t: t for t in tids}
    spans = {t: [tracks[t]["start"], tracks[t]["end"]] for t in tids}

    candidates = []
    for a in tids:
        for b in tids:
            if tracks[a]["end"] >= tracks[b]["start"]:
                continue  # not strictly earlier
            gap = tracks[b]["start"] - tracks[a]["end"]
            if gap > max_gap_frames:
                continue
            sim = float(np.dot(tracks[a]["embedding"], tracks[b]["embedding"]))
            if sim >= sim_threshold:
                candidates.append((sim, a, b))

    # Partner lists per end, for the ambiguity test below.
    partners: dict[int, list[tuple[float, int]]] = {}
    for sim, a, b in candidates:
        partners.setdefault(a, []).append((sim, b))
        partners.setdefault(b, []).append((sim, a))
    for plist in partners.values():
        plist.sort(reverse=True)

    def _ambiguous(end: int, sim: float, other: int) -> bool:
        """True when a rival partner is nearly as similar AND cannot be the same
        player as the proposed one. A rival that resembles the proposed partner
        and does not coexist with it in time is plausibly a further fragment of
        the same player - chaining through it later is safe, so it does not
        block the merge. A rival that overlaps the proposed partner in time is
        provably a different player (a same-kit teammate): genuine ambiguity."""
        for rival_sim, rival in partners[end]:
            if rival == other or rival_sim < sim - margin:
                continue
            coexist = not (
                tracks[rival]["end"] < tracks[other]["start"]
                or tracks[other]["end"] < tracks[rival]["start"]
            )
            rival_vs_other = float(np.dot(tracks[rival]["embedding"], tracks[other]["embedding"]))
            if coexist or rival_vs_other < sim_threshold:
                return True
        return False

    for sim, a, b in sorted(candidates, reverse=True):
        if _ambiguous(a, sim, b) or _ambiguous(b, sim, a):
            continue
        ra, rb = _find(parent, a), _find(parent, b)
        if ra == rb:
            continue
        # Chains must stay time-consistent: no overlap after the merge.
        sa, sb = spans[ra], spans[rb]
        if not (sa[1] < sb[0] or sb[1] < sa[0]):
            continue
        parent[rb] = ra
        spans[ra] = [min(sa[0], sb[0]), max(sa[1], sb[1])]

    return {t: _find(parent, t) for t in tids}
