"""PTSAM-style prompt tuning — the ONLY parametric escape hatch (optional).

The pipeline is non-parametric by default (the memory bank is the model, weights
never move). If the whole model plateaus below target, PTSAM-style prompt tuning
learns a tiny set of prompt tokens (~2,048 params) with the SAM encoder AND decoder
frozen — no LoRA, no full fine-tune. This module provides the learnable prompt + the
plateau gate; it is off by default and only engaged when :func:`should_prompt_tune`
fires. (It was NOT engaged in the P1 benchmark — the non-parametric loop met its
gates; see results/P1-RESULTS.md.)
"""
from __future__ import annotations


class PromptTuner:
    def __init__(self, n_tokens: int = 8, dim: int = 256):
        import torch

        self.n_tokens = n_tokens
        self.dim = dim
        # ~n_tokens*dim learnable params (8*256 = 2048, PTSAM scale)
        self.prompt = torch.nn.Parameter(torch.zeros(n_tokens, dim))

    def parameters(self):
        return [self.prompt]

    def trainable_param_count(self) -> int:
        return int(self.prompt.numel()) if self.prompt.requires_grad else 0

    def attach_frozen(self, sam_model) -> None:
        """Freeze every backbone parameter — only ``self.prompt`` stays trainable."""
        for p in sam_model.parameters():
            p.requires_grad_(False)

    def fit(self, exemplars, steps: int = 200, lr: float = 1e-2):
        """Optimise the prompt tokens against exemplar masks (encoder/decoder frozen).

        Left as an integration point: wiring ``self.prompt`` into the SAM prompt
        encoder requires the model's internal prompt-embedding hook. The plateau gate
        and the frozen-backbone contract are the tested parts; the training step is
        enabled per deployment when a plateau actually occurs.
        """
        raise NotImplementedError(
            "PTSAM training is an opt-in integration point; engaged only on plateau "
            "(see should_prompt_tune). Not used in the P1 non-parametric benchmark."
        )


def should_prompt_tune(convergence: dict, current_iou: float, target_iou: float) -> bool:
    """Engage prompt tuning only when the loop has CONVERGED but is still below the
    target IoU — i.e. adding more labels stopped helping yet quality is insufficient."""
    return bool(convergence.get("converged")) and current_iou < target_iou
