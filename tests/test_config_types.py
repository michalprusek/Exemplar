import numpy as np

from active_segmenter.config import RunConfig, EncoderConfig
from active_segmenter.types import InstanceMask, ClassLabel, ALState, MemoryBankEntry


def test_runconfig_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "device: cpu\nseed: 1\ncache_dir: /tmp/cache\nencoder:\n  resolution: 448\n"
    )
    cfg = RunConfig.from_yaml(str(p))
    assert cfg.device == "cpu"
    assert cfg.seed == 1
    assert cfg.encoder.resolution == 448
    assert cfg.match.topk == 5  # default preserved when yaml omits it


def test_device_auto_resolves_to_cpu(monkeypatch):
    import active_segmenter.config as C

    monkeypatch.setattr(C, "_cuda_available", lambda: False)
    monkeypatch.setattr(C, "_mps_available", lambda: False)
    cfg = RunConfig(device="auto", cache_dir="/tmp", encoder=EncoderConfig())
    assert cfg.device_resolved() == "cpu"


def test_device_auto_prefers_cuda(monkeypatch):
    import active_segmenter.config as C

    monkeypatch.setattr(C, "_cuda_available", lambda: True)
    monkeypatch.setattr(C, "_mps_available", lambda: False)
    cfg = RunConfig(device="auto", cache_dir="/tmp", encoder=EncoderConfig())
    assert cfg.device_resolved() == "cuda"


def test_explicit_device_passthrough():
    cfg = RunConfig(device="cpu", cache_dir="/tmp", encoder=EncoderConfig())
    assert cfg.device_resolved() == "cpu"


def test_instance_mask_fields():
    m = InstanceMask(mask=np.zeros((4, 4), bool), points=None, class_id=1, instance_id=2)
    assert m.class_id == 1 and m.instance_id == 2
    assert m.mask.shape == (4, 4)
    assert m.score == 1.0


def test_class_label():
    c = ClassLabel(id=3, name="nucleus", color="#ff0000")
    assert c.name == "nucleus"


def test_membank_entry():
    e = MemoryBankEntry(
        dataset_id="d", class_id=1, instance_id=2,
        embedding=np.zeros((2, 4), np.float32),
        exemplar_mask=np.zeros((4, 4), bool), round=0,
    )
    assert e.embedding.shape == (2, 4)


def test_alstate_roundtrip():
    st = ALState(round=3, selected=[1, 2], scores={"a": 0.1}, convergence={}, bank_size=5)
    js = st.to_json()
    back = ALState.from_json(js)
    assert back.round == 3
    assert back.selected == [1, 2]
    assert back.scores == {"a": 0.1}
