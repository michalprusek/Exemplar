"""Active-learning acquisition.

Cold start (TypiClust/ProbCover) seeds the bank without any uncertainty signal;
then rounds score the pool by EPIG-style generalisation-targeted acquisition over
cheap non-parametric uncertainty, batch-diversified with k-center/BADGE, always
against a random control arm. Convergence is a composite stopping signal.
"""
from active_segmenter.acquire import coldstart, convergence, diversity
from active_segmenter.acquire.base import AcqContext, Acquisition
from active_segmenter.acquire.epig import EpigAcq
from active_segmenter.acquire.random import RandomAcq
from active_segmenter.acquire.uncertainty import UncertaintyAcq

__all__ = [
    "AcqContext", "Acquisition", "EpigAcq", "RandomAcq", "UncertaintyAcq",
    "coldstart", "convergence", "diversity",
]


def build_acquisition(strategy: str) -> Acquisition:
    if strategy == "random":
        return RandomAcq()
    if strategy == "uncertainty":
        return UncertaintyAcq()
    if strategy == "epig":
        return EpigAcq()
    if strategy == "typiclust":
        from active_segmenter.acquire.typiclust import TypiClustAcq

        return TypiClustAcq()
    raise ValueError(f"unknown acquisition strategy: {strategy}")
