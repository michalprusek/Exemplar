"""Foreground-patch coverage — the acquisition's on-sphere diversity signal.

Classic cold-start AL (TypiClust/ProbCover) covers the pool over GLOBAL image (CLS)
descriptors. But an in-context segmenter fails at the PATCH level: a query foreground patch
gets mislabelled when no *foreground* support patch lies near it on the sphere. So coverage
should be computed over foreground-candidate PATCH features, not CLS — a candidate is
valuable when its foreground patches cover a region of the concept manifold the support bank
does not yet cover. This is the finer, patch-level refinement of SMEC's CLS-level coverage
(``[[smec-acquisition]]``); it needs a foreground hypothesis per pool image (from the current
bank's correspondence at query time), and degrades gracefully to CLS typicality when the bank
is empty (no fg hypothesis yet).

Pure, model-free functions (features injected) so they are unit-testable without the encoder,
matching the ``smec.py`` style. All feature arrays are assumed L2-normalised (cosine = dot).
"""
from __future__ import annotations

import numpy as np


def fg_patch_novelty(cand_fg_feats, bank_fg_feats) -> float:
    """Mean over the candidate's foreground patches of ``1 - max cosine to any bank fg
    patch``. 0 = every candidate fg patch is already covered by the bank; 1 = all lie in
    regions the bank's foreground never reaches. Empty bank fg ⇒ 1 (all novel); empty
    candidate fg ⇒ 0 (nothing to offer)."""
    cand = np.asarray(cand_fg_feats, np.float32)
    if cand.ndim != 2 or cand.shape[0] == 0:
        return 0.0
    bank = np.asarray(bank_fg_feats, np.float32)
    if bank.ndim != 2 or bank.shape[0] == 0:
        return 1.0
    sims = cand @ bank.T                       # [Nc, Nb] cosine (unit vectors)
    nearest = sims.max(axis=1)                  # closest bank fg patch per candidate patch
    return float(np.mean(np.maximum(0.0, 1.0 - nearest)))


def fg_coverage_scores(cand_fg_feats: dict, bank_fg_feats) -> dict:
    """``{pool_index: fg_patch_novelty(candidate, bank)}`` — higher = annotate first.

    ``cand_fg_feats`` maps each pool index to its ``[Nc, D]`` foreground-candidate patch
    features (from the bank's correspondence ``score_map > thresh`` at query time)."""
    bank = np.asarray(bank_fg_feats, np.float32)
    return {k: fg_patch_novelty(v, bank) for k, v in cand_fg_feats.items()}
