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


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020): a per-image soft weight on every
    channel, from a 1-D convolution over the globally-pooled channel descriptor.

    Why this shape of channel combination and not another. C36 established that HARD, support-driven
    Fisher selection of channels regresses every dataset, because channels are jointly useful and a
    univariate criterion cannot see that; the 1x1's soft per-channel weighting beat it. ECA is the
    same soft idea taken further: it re-weights channels PER QUERY IMAGE rather than once for the
    whole run, and it does so without the bottleneck that Squeeze-and-Excitation needs (a 1024-channel
    SE block at r=16 costs 131,072 parameters; ECA costs k, here 5). That matters because this head
    trains on eight masks, where 131k extra parameters is not a small ask and 5 is free.

    It is also complementary to what the head already conditions on, rather than a second copy of it:
    FiLM scales channels from the SUPPORT prototypes, so it adapts to the dataset and is identical
    for every query; ECA scales them from the QUERY's own pooled response, so it adapts per image.

    Parity at init: the weights are zero-initialised and the gate is 2*sigmoid, so it starts at
    exactly 1.0 and can only depart from identity by learning -- the same convention as the
    competitive gate, the FiLM hypernetwork and the guided upsampler.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        import math
        t = int(abs((math.log2(max(channels, 2)) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        y = x.mean(dim=(2, 3))                                  # [B, C] pooled channel descriptor
        y = self.conv(y.unsqueeze(1)).squeeze(1)                # [B, C] local cross-channel mixing
        return x * (2.0 * torch.sigmoid(y)).unsqueeze(-1).unsqueeze(-1)



class SE(nn.Module):
    """Squeeze-and-Excitation (Hu et al., CVPR 2018): per-image channel weights through a bottleneck.

    Chosen over ECA specifically because it is PERMUTATION-INVARIANT over channels. ECA weights each
    channel from a 1-D convolution over its index neighbours, which assumes adjacent channel indices
    are related; on cached DINOv3 features adjacent channels correlate 0.1685 against 0.1714 for
    distant pairs (ratio 0.98), so that assumption is void for a frozen transformer and ECA regressed
    (C41). SE's two dense layers see every channel, so channel order cannot mislead it.

    The price is parameters, which is the axis this head is most sensitive to at K=8: 2*C*C/r, i.e.
    131,072 at C=1024 and the usual r=16, against ECA's 5. `reduction` is therefore exposed rather
    than fixed, so the screen can ask whether the capacity helps or repeats the stem's overfitting.

    Parity at init: the second projection is zero-initialised and the gate is 2*sigmoid, so it starts
    at exactly 1.0 -- the convention every lever here follows.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        y = x.mean(dim=(2, 3))                                  # [B, C] squeeze
        y = self.fc2(torch.nn.functional.gelu(self.fc1(y)))     # [B, C] excite
        return x * (2.0 * torch.sigmoid(y)).unsqueeze(-1).unsqueeze(-1)



class Bilinear(nn.Module):
    """Low-rank bilinear interaction between the fused sources (MLB/MFB family).

    Concatenation followed by a 1x1 classifier is ADDITIVE: the logit is a weighted sum over
    channels, so it cannot represent "DINO says vessel AND the Frangi ridge is high" differently
    from the sum of the two. For this task that coincidence is exactly the reliable signal -- the
    classical priors alone fire on any edge, the backbone alone gives a diffuse boundary, and it is
    their agreement that localises. The competitive gate supplies only a per-GROUP scalar, which can
    say "trust the priors here" but cannot form a channel-level product.

    Full bilinear pooling over three sources is out of the question (its dimensionality is quadratic
    in the channel count). The low-rank form projects each source to `rank` and takes Hadamard
    products of the pairs, which costs rank*(sum of source widths) parameters -- 1,584 at rank 16 --
    and appends 3*rank channels carrying the interactions the linear head cannot otherwise express.

    Zero-init on the projections would kill the gradient, so parity is achieved instead by appending
    the products as EXTRA channels: the classifier's weights on them start at zero, so the module is
    exactly inert at init while its inputs still receive gradient.
    """

    def __init__(self, widths, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.proj = nn.ModuleList([nn.Conv2d(w, rank, 1, bias=False) for w in widths])
        self.n_out = rank * (len(widths) * (len(widths) - 1) // 2)

    def forward(self, groups):
        if len(groups) != len(self.proj):
            raise ValueError(f"Bilinear built for {len(self.proj)} sources but got {len(groups)}; "
                             f"zip would truncate and emit the wrong channel count silently")
        p = [f(g) for f, g in zip(self.proj, groups)]
        out = []
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                out.append(p[i] * p[j])                      # Hadamard = the interaction term
        return torch.cat(out, dim=1)


class DINOHeadFusion(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 256, n_classes: int = 1,
                 proj_dim: int = 32, n_classical: int = 35, guided_fuse: bool = False,
                 boundary_head: bool = False, scale_fusion: bool = False, upsampler=None,
                 dist_head: bool = False, competitive_gate: bool = False, corr_prior: bool = False,
                 film: bool = False, hidden_film: int = 64, stem: str = "wide",
                 depth: int = 2, eca: bool = False, dropout: float = 0.0,
                 se: int = 0, mix: bool = False, bilinear: int = 0):
        super().__init__()
        groups = min(8, hidden)
        # STEM WIDTH/COST. The first convolution does two jobs at once: it mixes 3x3 spatially AND
        # reduces 1024 backbone channels to ``hidden``. Paying the 9x kernel multiplier on the WIDEST
        # channel transition is what makes it 1024*256*9 = 2.36M parameters, 76% of the whole head,
        # while the second convolution at 256 costs a quarter of that for the same receptive field.
        #
        #   "wide" (default) -- 3x3 reduce, then 3x3. What every reported number was produced with.
        #   "lean"           -- 1x1 reduce, then 3x3. The spatial mixing moves to the cheap width; the
        #                       grid-resolution receptive field is unchanged (the second conv still
        #                       sees 3x3), and DINOv3's features are already contextual, so the first
        #                       layer has little spatial work to do that attention has not done.
        #   "flat"           -- 1x1, then 1x1. No spatial convolution at all: a per-position channel
        #                       mixer. This asks whether the head needs ANY grid-resolution spatial
        #                       context of its own, given that self-attention has already mixed
        #                       globally and the upsampler re-introduces spatial structure from the
        #                       native classical priors afterwards.
        #
        # Combined with ``hidden``, "lean" spans roughly 3.10M -> 0.43M without touching anything else.
        # Default is "wide": switching it changes every number in the paper and must be A/B'd first.
        if stem not in ("wide", "lean", "flat"):
            raise ValueError(f"unknown stem {stem!r}; expected 'wide', 'lean' or 'flat'")
        self.stem = stem
        k1 = 3 if stem == "wide" else 1
        k2 = 1 if stem == "flat" else 3
        # DEPTH. Two hidden layers is what every reported number used. One is enough to keep the map
        # from raw features NONLINEAR, which is the property the C26 probe found to matter (a linear
        # rule on frozen features loses ~0.06 fg-IoU to an MLP); the second layer adds capacity, not
        # expressiveness, and capacity is what overfits on eight training images.
        if depth not in (1, 2):
            raise ValueError(f"stem depth must be 1 or 2, got {depth!r}")
        self.depth = depth
        # DROPOUT. Dropout2d, not plain Dropout: these are convolutional features, where adjacent
        # positions in a channel are strongly correlated, so dropping individual elements mostly
        # adds noise the neighbours reconstruct. Dropping whole CHANNELS is the version that
        # actually regularises a conv stack. Applied after each activation, and inactive at eval
        # through the module's own train/eval flag, which the backend already toggles.
        def _do():
            return [nn.Dropout2d(dropout)] if dropout > 0 else []
        layers = [nn.Conv2d(in_dim, hidden, k1, padding=k1 // 2),
                  nn.GroupNorm(groups, hidden), nn.GELU()] + _do()
        if depth == 2:
            layers += [nn.Conv2d(hidden, hidden, k2, padding=k2 // 2),
                       nn.GroupNorm(groups, hidden), nn.GELU()] + _do()
        self.body = nn.Sequential(*layers)
        self.dropout = dropout
        # widths of the concatenated groups: coarse [+ fine] + classical [+ corr]
        _w = [proj_dim] + ([proj_dim] if scale_fusion else []) + [n_classical + (1 if corr_prior else 0)]
        self.bil = Bilinear(_w, bilinear) if bilinear else None
        self.eca = ECA(in_dim) if eca else None
        self.se = SE(in_dim, se) if se else None
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
        # The bilinear block appends its interaction channels to the concatenation, so everything
        # downstream (FiLM, the mixing block, the classifier) must be built at the WIDER size. The
        # per-pixel gate is NOT: it still scores the original groups, and runs before the append.
        d_gate = d
        d = d + (self.bil.n_out if self.bil is not None else 0)
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
            self.gate = nn.Conv2d(d_gate, self.n_groups, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            self.gate_temp = nn.Parameter(torch.tensor(1.0))
        # Improvement C (guided fusion): a light separable block (depthwise 3×3 spatial context +
        # 1×1 channel mix) lets classical native structure GUIDE the upsampled semantics before the
        # classifier — sharper than concat+1×1. Cheap; keeps the final 1×1 classifier so EGL's
        # closed-form (p-y)·penultimate is unchanged.
        # POST-CONCAT MIXING. `guided_fuse` is the original form and carries a depthwise 3x3, which
        # the stem ladder measured as specifically harmful here. `mix` is the same idea without the
        # spatial kernel: one 1x1 plus GELU, which is what lets the classifier express products of
        # channels ACROSS sources instead of only their weighted sum.
        self.fuse = (nn.Sequential(nn.Conv2d(d, d, 3, padding=1, groups=d), nn.Conv2d(d, d, 1),
                                   nn.GELU()) if guided_fuse else
                     nn.Sequential(nn.Conv2d(d, d, 1), nn.GELU()) if mix else None)
        self.classifier = nn.Conv2d(d, n_classes, 1)                     # native-res fusion 1×1 (fg)
        if self.bil is not None:
            # PARITY AT INIT, which the Bilinear docstring promised and nothing delivered: the
            # classifier was default-initialised over the WIDENED width, so the appended interaction
            # channels perturbed the logit at step 0 and a `bil` A/B would have confounded the lever
            # with a different classifier init. Zeroing only the appended columns leaves the module
            # exactly inert while its projections still receive gradient -- which is why the products
            # are appended rather than the projections zero-initialised.
            with torch.no_grad():
                self.classifier.weight[:, -self.bil.n_out:].zero_()
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
        if self.eca is not None:
            grid_bchw = self.eca(grid_bchw)      # soft per-image channel re-weighting
        if self.se is not None:
            grid_bchw = self.se(grid_bchw)       # ditto, permutation-invariant over channels
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
        if self.bil is not None:
            groups = groups + [self.bil(groups)]   # + cross-source interaction channels
        z = torch.cat(groups, dim=1)                                     # coarse[⊕fine]⊕classical[⊕products]
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
