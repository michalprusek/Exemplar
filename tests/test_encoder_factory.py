from active_segmenter.config import EncoderConfig
from active_segmenter.encoder.factory import cache_tag, is_convnext


def test_is_convnext_infers_from_model_id():
    vit = EncoderConfig(model_id="facebook/dinov3-vitl16-pretrain-lvd1689m")
    cnx = EncoderConfig(model_id="facebook/dinov3-convnext-large-pretrain-lvd1689m")
    assert is_convnext(vit) is False
    assert is_convnext(cnx) is True


def test_backbone_override_wins():
    cfg = EncoderConfig(model_id="something-vit", backbone="convnext")
    assert is_convnext(cfg) is True
    cfg2 = EncoderConfig(model_id="something-convnext", backbone="vit")
    assert is_convnext(cfg2) is False


def test_cache_tag_distinguishes_backbone_stage_res():
    vit = EncoderConfig(model_id="facebook/dinov3-vitl16-pretrain-lvd1689m", resolution=672)
    cnx = EncoderConfig(model_id="facebook/dinov3-convnext-large-pretrain-lvd1689m",
                        resolution=1024, convnext_stage=2)
    t_vit, t_cnx = cache_tag(vit), cache_tag(cnx)
    assert "res672" in t_vit and "cnxs" not in t_vit
    assert "res1024" in t_cnx and "cnxs2" in t_cnx
    assert t_vit != t_cnx


def test_native_cache_tag_and_vit_guard():
    import pytest
    from active_segmenter.encoder.factory import make_encoder

    cnx_nat = EncoderConfig(model_id="facebook/dinov3-convnext-large-pretrain-lvd1689m",
                            resolution=0, convnext_stage=2)
    assert "resNAT" in cache_tag(cnx_nat)
    # native resolution with a ViT backbone must be rejected (fixed patch grid)
    with pytest.raises(ValueError):
        make_encoder(EncoderConfig(model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
                                   resolution=0), "cpu")


def test_cache_tag_includes_layer_gram_tile():
    vit = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    base = cache_tag(EncoderConfig(model_id=vit, resolution=672))
    l18 = cache_tag(EncoderConfig(model_id=vit, resolution=672, layer=18))
    gram = cache_tag(EncoderConfig(model_id=vit, resolution=672, gram_refine=True))
    tiled = cache_tag(EncoderConfig(model_id=vit, resolution=672, tile=True))
    tags = {base, l18, gram, tiled}
    assert len(tags) == 4  # every feature-affecting knob yields a distinct cache key
    assert "L18" in l18 and "gram" in gram and "tiled" in tiled
