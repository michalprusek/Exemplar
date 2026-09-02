"""Configuration dataclasses for the active-segmenter pipeline.

Everything the pipeline needs to run is captured here so a benchmark can be
reproduced from a single YAML file plus a git SHA. Device resolution is
centralised: code never hard-codes ``"cuda"`` — it asks the config, which can
auto-pick ``cuda`` > ``mps`` > ``cpu``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import yaml


# -- device probes (monkeypatchable in tests) --------------------------------
def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:
        return False


@dataclass
class EncoderConfig:
    model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    resolution: int = 672  # multiple of 16 -> resolution//16 patch grid
    tile: bool = False
    tile_overlap: float = 0.25
    project_pos_bias: bool = False
    feat_dim: int = 1024
    patch_stride: int = 16
    n_prefix_tokens: int = 5  # CLS + 4 register tokens to drop
    # FSSDINO (2602.07550) "semantic selection gap": intermediate DINOv3 ViT layers beat the
    # last by 6-13 mIoU on fine structures, but no heuristic reliably picks the layer -> expose
    # it. layer == -1 = last_hidden_state (default); else hidden_states[layer]. gram_refine adds
    # one step of Gram (patch self-similarity) propagation to sharpen dense correspondence.
    layer: int = -1
    # LAYER-FUSION: non-empty = concat these DINOv3 block outputs along the feature axis (e.g. (-1, 12)) and
    # let the light head learn the per-layer weighting from the K labels (FSSDINO: no heuristic reliably picks
    # the layer). Empty = single ``layer``. A universal fixed set for ALL datasets; the per-dataset adaptation
    # is in the trainable head. MUST enter cache_tag (feature-affecting).
    layers: tuple = ()
    gram_refine: bool = False
    # -- training-free feature super-resolution (encoder/superres.py) ------------
    # superres_factor>1 densifies the ViT patch grid factor× via sub-patch shift-merge
    # (factor² forward passes). jbu edge-snaps the finer grid (parameter-free joint bilateral,
    # a feature-level CRF). Alternative to `tile`, NOT stacked. BOTH must enter cache_tag.
    superres_factor: int = 1
    jbu: bool = False
    jbu_sigma_spatial: float = 1.0
    jbu_sigma_range: float = 0.1
    # -- learned frozen-feature upsampler (encoder/feat_upsample.py) --------------
    # The LABEL-FREE alternative to shift-merge superres: a pretrained upsampler (AnyUp ICLR'26 /
    # JAFAR NeurIPS'25) lifts the coarse /16 grid to factor× finer in ONE feed-forward pass, guided
    # by the RGB image. Unlike superres (factor² REAL forward passes = genuinely new sub-patch signal)
    # it RECOVERS high-freq detail from the guide but adds no sub-patch info absent from the /16 pass.
    # "none" (default) | "anyup" | "jafar". MUTUALLY EXCLUSIVE with superres_factor; BOTH the name and
    # factor MUST enter cache_tag (a silent collision poisons results — cf. the HRF layer-sweep bug).
    feat_upsampler: str = "none"
    feat_upsample_factor: int = 2   # target grid = factor × base grid (match superres for a fair A/B)
    # -- convolutional backbone (DINOv3-ConvNeXt) for fine full-res grids --------
    # ViT patch-16 gives a coarse resolution//16 grid; a ConvNeXt backbone gives a
    # hierarchical feature pyramid, so small objects survive. Selected when model_id
    # contains "convnext" (or backbone == "convnext"). convnext_stage indexes the HF
    # hidden_states: 1=stride4, 2=stride8, 3=stride16, 4=stride32 (finer = smaller stride).
    backbone: str = "auto"          # "auto" (infer from model_id) | "vit" | "convnext"
    convnext_stage: int = 2         # stride-8 grid (resolution//8) — fine but semantic
    # resolution <= 0 with a convnext backbone = NATIVE: each image is fed at its own H×W
    # (snapped to a multiple of 32, capped) so the conv trunk segments at native resolution.


@dataclass
class MatchConfig:
    topk: int = 5
    bidirectional: bool = True
    fg_bg_margin_eps: float = 0.03  # |score| below this = ambiguous patch
    max_fg: int = 8000   # cap fg exemplar patches per match (subsample if exceeded)
    max_bg: int = 16000  # cap bg exemplar patches per match


@dataclass
class ClusterConfig:
    algo: str = "agglomerative"  # or "hdbscan"
    xy_weight: float = 0.3       # (reserved) spatial vs feature balance
    use_features: bool = False   # cluster on xy only by default; features optional
    feature_gain: float = 1.5    # scale on appended features when use_features
    score_thresh: float = 0.0
    distance_threshold: float = 1.5
    max_instances: int = 256
    min_patches: int = 2


@dataclass
class RefineConfig:
    kind: str = "identity"  # "identity" | "sam"
    sam_id: str = "facebook/sam2.1-hiera-large"
    use_crf: bool = False
    sam_negatives: bool = True  # sibling negatives SEPARATE touching instances;
    #                             set False to PRESERVE amodal overlap (masks may share pixels)
    # -- prompt strength (refine/sam.py) -----------------------------------------
    # point: single interior point (current). mask: also feed the coarse proposal mask as SAM's
    # low-res mask prompt (carries shape -> tighter boundaries). mask_box: mask + bbox prompt.
    prompt_mode: str = "point"   # "point" | "mask" | "mask_box"
    # amodal: keep overlapping per-instance masks (no sibling negatives, no cross-suppression).
    amodal: bool = False


@dataclass
class AcquireConfig:
    strategy: str = "typiclust"  # benchmarked winner; also "epig" | "uncertainty" | "random"
    coldstart: str = "typiclust"  # "typiclust" | "probcover"
    topk_batch: int = 1
    bank_cap: int = 64
    diversity: str = "kcenter"   # "kcenter" | "badge" | "none"


@dataclass
class RunConfig:
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    seed: int = 0
    cache_dir: str = "/tmp/active_segmenter_cache"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    acquire: AcquireConfig = field(default_factory=AcquireConfig)

    def device_resolved(self) -> str:
        if self.device != "auto":
            return self.device
        if _cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    # -- (de)serialisation ---------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        return _build(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(dc_type, data: dict[str, Any]):
    """Recursively build a (possibly nested) dataclass, overlaying ``data`` on
    top of each field's default so partial YAML still yields a full config."""
    kwargs: dict[str, Any] = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        val = data[f.name]
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[f.name] = _build(f.type, val)
        elif isinstance(val, dict) and _is_nested_config(f):
            kwargs[f.name] = _build(_field_default_type(f), val)
        else:
            kwargs[f.name] = val
    return dc_type(**kwargs)


def _is_nested_config(f) -> bool:
    return f.default_factory is not dataclasses.MISSING and is_dataclass(
        f.default_factory
    )


def _field_default_type(f):
    # default_factory is the dataclass type itself (e.g. EncoderConfig)
    return f.default_factory
