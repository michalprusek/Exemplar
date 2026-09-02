"""RefineConfig prompt-mode + amodal knobs (spec 2026-07-12 refine-stage)."""
from active_segmenter.config import RefineConfig, RunConfig


def test_refine_config_defaults_backward_compatible():
    c = RefineConfig()
    assert c.prompt_mode == "point" and c.amodal is False
    assert c.kind == "identity" and c.sam_negatives is True


def test_refine_config_from_yaml_roundtrip(tmp_path):
    import yaml

    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"refine": {"kind": "sam", "prompt_mode": "mask", "amodal": True}}))
    cfg = RunConfig.from_yaml(str(p))
    assert cfg.refine.prompt_mode == "mask" and cfg.refine.amodal is True
