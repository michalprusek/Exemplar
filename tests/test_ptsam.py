import numpy as np

from active_segmenter.ptsam.prompt_tuning import PromptTuner, should_prompt_tune


def test_prompt_has_about_2048_trainable_params():
    t = PromptTuner(n_tokens=8, dim=256)
    assert t.trainable_param_count() == 8 * 256  # 2048, PTSAM-scale


def test_freeze_backbone_leaves_only_prompt_trainable():
    import torch

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Linear(4, 4)  # stand-in for SAM encoder/decoder

    sam = TinyBackbone()
    t = PromptTuner(n_tokens=4, dim=16)
    t.attach_frozen(sam)
    # backbone params frozen, only the prompt is trainable
    assert all(not p.requires_grad for p in sam.parameters())
    trainable = [p for p in t.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == 4 * 16


def test_should_prompt_tune_only_on_plateau_below_target():
    # converged but IoU below target -> prompt-tune
    conv = {"converged": True}
    assert should_prompt_tune(conv, current_iou=0.55, target_iou=0.70) is True
    # converged and already at target -> no
    assert should_prompt_tune(conv, current_iou=0.75, target_iou=0.70) is False
    # not converged -> no
    assert should_prompt_tune({"converged": False}, current_iou=0.5, target_iou=0.7) is False
