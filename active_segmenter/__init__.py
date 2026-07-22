"""In-context active-learning polygon segmenter (P1 ML library).

Frozen DINOv3-L propose -> per-patch kNN correspondence -> clustering instance
decomposition (overlap-preserving) -> pluggable SAM refine -> polygon, driven by
an active-learning correct-and-advance loop against a random control arm.
"""

__version__ = "0.1.0"
