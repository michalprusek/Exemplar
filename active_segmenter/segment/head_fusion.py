"""Native-resolution fusion head: DINOv3 semantics + classical per-pixel priors.

Extends :class:`DINOHead` so the *existing* pipeline (few-shot fit, EGL/BADGE acquisition) keeps
working, but the decision is made at NATIVE resolution with classical per-pixel features added —
the small-blob detail a patch-16 grid discards. The DINO body runs on the coarse grid (semantics),
is projected to a small ``proj_dim`` and upsampled to native, then concatenated with the classical
feature stack ``[H, W, C]``; a 1×1 classifier fuses them per native pixel. Because the classifier is
1×1, the weight-coupled acquisition gradient ``(sigmoid(logit) - y) · penultimate`` is unchanged —
just over a richer (semantic ⊕ classical) penultimate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHeadFusion(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 256, n_classes: int = 1,
                 proj_dim: int = 32, n_classical: int = 35, guided_fuse: bool = False,
                 boundary_head: bool = False, scale_fusion: bool = False, upsampler=None,
                 dist_head: bool = False, competitive_gate: bool = False, corr_prior: bool = False,
                 film: bool = False, hidden_film: int = 64):
        super().__init__()
        groups = min(8, hidden)
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1), nn.GroupNorm(groups, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(groups, hidden), nn.GELU(),
        )
        self.proj = nn.Conv2d(hidden, proj_dim, 1)                       # semantic embedding, small
        # CONTINUOUS scale-fusion (replaces the discrete crop-vs-whole gate): the SAME body+proj
        # embeds two DINOv3 scales — coarse (whole-image context) and fine (native-tiled detail) —
        # which are simply CONCATENATED with the classical priors; the 1×1 fuse/classifier learns the
        # per-channel weighting (continuous & learnable, no explicit gate). The scale balance adapts to
        # the dataset from the few labels (thin structures lean on fine, blobs on coarse) via the mix.
        self.scale_fusion = scale_fusion
        # learned feature upsampler (replaces bilinear when set) — lifts coarse/fine embeddings to the
        # classical native res, optionally GUIDED by the native classical priors (edges/ridges).
        from active_segmenter.segment.upsamplers import make_upsampler
        self.up = make_upsampler(upsampler, proj_dim, guide_ch=n_classical)
        # CORRESPONDENCE PRIOR: a support-derived per-pixel "how much does this look like the support
        # FOREGROUND vs background" channel — cosine(feat, fg_proto) − cosine(feat, bg_proto), appended to the
        # classical native priors. Self-configuring (prototypes from the K support masks), directly targets the
        # foreground (the instance-AP bottleneck per the oracle-fg diagnosis). Prototypes set post-fit.
        self.corr_prior = corr_prior
        self.register_buffer("fg_proto", None, persistent=False)
        self.register_buffer("bg_proto", None, persistent=False)
        # MULTI-PROTOTYPE (Lever 1): optional k-means centroid stacks ([k,D]/[j,D], unit rows). When set,
        # the corr channel is max_k cos(feat,fg_k) − max_j cos(feat,bg_j) (one channel, width unchanged);
        # None → the single-prototype channel above (parity). Set by set_prototypes from the backend.
        self.register_buffer("fg_protos", None, persistent=False)
        self.register_buffer("bg_protos", None, persistent=False)
        d = (2 * proj_dim if scale_fusion else proj_dim) + n_classical + (1 if corr_prior else 0)
        self.d = d                                                       # penultimate width (= classifier in_channels)
        # SUPPORT-CONDITIONED FiLM: a tiny hypernetwork maps the SUPPORT summary s = [fg_proto ⊕ bg_proto]
        # (the SAME prototypes the corr_prior uses, [2D]) → per-channel (γ, β) that modulate the penultimate
        # z BEFORE the 1×1 classifier, so the fusion adapts to the dataset from the K support masks. The final
        # Linear is ZERO-INIT (weight AND bias) → γ=1+0=1, β=0 at init → an EXACT identity → starts EXACTLY at
        # the un-modulated (current best) head and only sharpens from the K labels (same parity rationale as the
        # competitive gate's zero-init). If prototypes are None/degenerate, FiLM is skipped (identity). The final
        # classifier stays a 1×1 so EGL's closed-form (p−y)·penultimate is unchanged.
        self.film = film
        self.film_net = None
        if film:
            self.film_net = nn.Sequential(
                nn.Linear(2 * in_dim, hidden_film), nn.GELU(), nn.Linear(hidden_film, 2 * d),
            )
            nn.init.zeros_(self.film_net[-1].weight)                     # parity at init: γ=1, β=0 → identity
            nn.init.zeros_(self.film_net[-1].bias)
        # COMPETITIVE GATE: per-pixel softmax over the filter GROUPS (coarse[+fine] DINO, classical) so a
        # CONFIDENT group DOMINATES the others (winner-take-more) instead of a static linear sum. Zero-init +
        # learnable temperature → starts EXACTLY at the uniform (current) weighting and sharpens from the K
        # labels; ``w = G·softmax(gate/τ)`` sums to G, so each group starts at 1.0 (parity) and one can take most.
        self.n_groups = (2 if scale_fusion else 1) + 1
        self.gate = None
        if competitive_gate:
            self.gate = nn.Conv2d(d, self.n_groups, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            self.gate_temp = nn.Parameter(torch.tensor(1.0))
        # Improvement C (guided fusion): a light separable block (depthwise 3×3 spatial context +
        # 1×1 channel mix) lets classical native structure GUIDE the upsampled semantics before the
        # classifier — sharper than concat+1×1. Cheap; keeps the final 1×1 classifier so EGL's
        # closed-form (p-y)·penultimate is unchanged.
        self.fuse = (nn.Sequential(nn.Conv2d(d, d, 3, padding=1, groups=d), nn.Conv2d(d, d, 1),
                                   nn.GELU()) if guided_fuse else None)
        self.classifier = nn.Conv2d(d, n_classes, 1)                     # native-res fusion 1×1 (fg)
        # W2 (learned instance separation): a 2nd 1×1 head predicts inter-instance BOUNDARIES so
        # touching objects stop merging in the mask (subtract at inference → watershed). Separate
        # head → the fg classifier (and EGL's closed-form over it) is unchanged.
        self.boundary = nn.Conv2d(d, 1, 1) if boundary_head else None
        # DT-regression instance head (StarDist/micro-SAM-AIS pattern): predicts a per-instance normalised
        # CENTER-DISTANCE map (1 at each instance's centre → 0 at its boundary). Its local maxima are ONE
        # marker per instance for a seeded watershed — smoother + far less few-shot-hungry than the boundary
        # CLASSIFICATION head (which NULLed few-shot). Separate 1×1 head → the fg classifier is unchanged.
        self.dist = nn.Conv2d(d, 1, 1) if dist_head else None

    def set_prototypes(self, fg_proto, bg_proto, fg_protos=None, bg_protos=None):
        """Store the support-derived unit fg/bg feature prototypes ([D] each) for the correspondence prior AND
        the FiLM conditioning vector s=[fg_proto⊕bg_proto]; called by the backend after fit builds them from
        the K support masks. Optional ``fg_protos``/``bg_protos`` ([k,D]/[j,D] unit rows) enable the
        multi-prototype MAX-POOLED corr channel; passing None (default) keeps the single-prototype channel
        (parity). FiLM always conditions on the mean ``fg_proto``/``bg_proto`` (unchanged)."""
        dev = self.classifier.weight.device
        self.fg_proto = fg_proto.float().to(dev)
        self.bg_proto = bg_proto.float().to(dev)
        self.fg_protos = fg_protos.float().to(dev) if fg_protos is not None else None
        self.bg_protos = bg_protos.float().to(dev) if bg_protos is not None else None

    def _embed(self, grid_bchw, out_hw, guide=None):
        h = self.proj(self.body(grid_bchw))                              # [1, proj, g, g]
        if self.up is not None:                                          # learned upsampler
            return self.up(h, out_hw, guide)
        return F.interpolate(h, size=out_hw, mode="bilinear", align_corners=False)  # → native

    def _penultimate(self, feat_grid_bchw, classical_bchw, out_hw, fine_grid_bchw=None):
        groups = [self._embed(feat_grid_bchw, out_hw, classical_bchw)]   # coarse (context)
        if self.scale_fusion:                                            # + fine (native detail)
            groups.append(self._embed(fine_grid_bchw, out_hw, classical_bchw)
                          if fine_grid_bchw is not None else groups[0])
        cls = classical_bchw
        if self.corr_prior and self.fg_proto is not None:                # + support correspondence-prior channel
            f = feat_grid_bchw.float()
            if self.fg_protos is not None and self.bg_protos is not None:  # multi-prototype: max_k cos − max_j cos
                # fp32-EXACT: under bf16 autocast the einsum lowers to bmm and would be demoted to bf16 (the
                # single-proto mul+sum stays fp32); disable autocast so the corr channel matches the fp32
                # pre-screen that gates this lever — mirrors the FiLM block's `enabled=False` guard.
                with torch.autocast(device_type=f.device.type, enabled=False):
                    fk = torch.einsum("bdhw,kd->bkhw", f, self.fg_protos.to(f))
                    bk = torch.einsum("bdhw,jd->bjhw", f, self.bg_protos.to(f))
                    corr = fk.amax(1, keepdim=True) - bk.amax(1, keepdim=True)         # [1,1,G,G]
            else:                                                         # single prototype: cos(feat,fg) − cos(feat,bg)
                corr = ((f * self.fg_proto.view(1, -1, 1, 1)).sum(1, keepdim=True)
                        - (f * self.bg_proto.view(1, -1, 1, 1)).sum(1, keepdim=True))  # [1,1,G,G]
            corr = F.interpolate(corr, size=out_hw, mode="bilinear", align_corners=False).to(cls.dtype)
            cls = torch.cat([cls, corr], dim=1)                          # [1, n_classical+1, H, W]
        groups.append(cls)                                               # + classical native priors [+ corr]
        if self.gate is not None:                                        # competitive: groups compete per pixel
            w = len(groups) * torch.softmax(self.gate(torch.cat(groups, dim=1))
                                            / self.gate_temp.clamp(min=0.05), dim=1)
            groups = [g * w[:, i:i + 1] for i, g in enumerate(groups)]   # confident group dominates by its weight
        z = torch.cat(groups, dim=1)                                     # coarse[⊕fine]⊕classical
        if self.film_net is not None and self.fg_proto is not None:      # support-conditioned FiLM (identity at init)
            # γ,β from the fixed fp32 prototypes through the (trainable) hypernet, in fp32 (autocast-disabled →
            # exact fp32 like corr_prior's fp32 corr channel), then cast to z.dtype before the per-channel affine.
            with torch.autocast(device_type=z.device.type, enabled=False):
                s = torch.cat([self.fg_proto, self.bg_proto], dim=0).float()   # [2D] support summary
                gb = self.film_net(s)                                          # [2d] fp32
            gamma = (1.0 + gb[:self.d]).to(z.dtype)                            # γ = 1 + γ_raw (identity at init)
            beta = gb[self.d:].to(z.dtype)                                     # β = β_raw (0 at init)
            z = gamma.view(1, self.d, 1, 1) * z + beta.view(1, self.d, 1, 1)
        return self.fuse(z) if self.fuse is not None else z

    def forward(self, feat_grid_bchw, classical_bchw, out_hw, fine_grid_bchw=None):
        return self.classifier(self._penultimate(feat_grid_bchw, classical_bchw, out_hw, fine_grid_bchw))

    def forward_with_penultimate(self, feat_grid_bchw, classical_bchw, out_hw, fine_grid_bchw=None):
        z = self._penultimate(feat_grid_bchw, classical_bchw, out_hw, fine_grid_bchw)
        return self.classifier(z), z

    def forward_fg_boundary(self, feat_grid_bchw, classical_bchw, out_hw, fine_grid_bchw=None):
        """(fg_logit, boundary_logit) — boundary_logit is None if no boundary head."""
        z = self._penultimate(feat_grid_bchw, classical_bchw, out_hw, fine_grid_bchw)
        b = self.boundary(z) if self.boundary is not None else None
        return self.classifier(z), b
