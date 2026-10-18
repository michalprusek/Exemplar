"""Exemplar: in-context biomedical segmentation by a closed-form readout of frozen inputs.

From a few annotated images, a linear readout of a frozen DINOv3 backbone (a coarse and a magnified fine view) and 35
classical priors is solved in closed form, with every choice (ridge weight, blend of two readouts) made on held-out
support images. No gradient step, nothing meta-trained. The method reported in the ISBI 2027 paper (v4).
"""
from exemplar.model import Exemplar, FrozenInputs

__version__ = "1.0.0"
__all__ = ["Exemplar", "FrozenInputs", "__version__"]
