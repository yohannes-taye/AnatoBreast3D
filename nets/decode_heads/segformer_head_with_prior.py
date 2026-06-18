"""SegFormer decode head variant that concatenates Gaussian priors."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from nets.segformer import ConvModule, MLP


class SegFormerHeadWithPrior(nn.Module):
    """SegFormer decode head that optionally consumes a 2-channel prior."""

    def __init__(
        self,
        num_classes: int,
        in_channels: Sequence[int],
        embedding_dim: int,
        dropout_ratio: float = 0.1,
        use_prior: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_prior = use_prior

        c1, c2, c3, c4 = in_channels
        self.linear_c4 = MLP(input_dim=c4, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1, embed_dim=embedding_dim)
        self.linear_fuse = ConvModule(c1=embedding_dim * 4, c2=embedding_dim, k=1)
        self.prior_adapter = None
        if use_prior:
            self.prior_adapter = nn.Sequential(
                nn.Conv2d(embedding_dim + 2, embedding_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, embedding_dim),
                nn.SiLU(inplace=True),
            )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, inputs: Sequence[Tensor], prior: Tensor | None = None) -> Tensor:
        """Decode fused transformer features with an optional prior."""
        c1, c2, c3, c4 = inputs
        batch_size = c1.shape[0]

        def _proj(feat: Tensor, projector: MLP) -> Tensor:
            proj = projector(feat).permute(0, 2, 1)
            return proj.reshape(batch_size, -1, feat.shape[2], feat.shape[3]).contiguous()

        _c4 = F.interpolate(_proj(c4, self.linear_c4), size=c1.shape[-2:], mode="bilinear", align_corners=False)
        _c3 = F.interpolate(_proj(c3, self.linear_c3), size=c1.shape[-2:], mode="bilinear", align_corners=False)
        _c2 = F.interpolate(_proj(c2, self.linear_c2), size=c1.shape[-2:], mode="bilinear", align_corners=False)
        _c1 = _proj(c1, self.linear_c1)

        fused = self.linear_fuse(torch.cat((_c4, _c3, _c2, _c1), dim=1))
        if self.use_prior and prior is not None and self.prior_adapter is not None:
            prior_resized = F.interpolate(prior, size=fused.shape[-2:], mode="bilinear", align_corners=False)
            fused = self.prior_adapter(torch.cat((fused, prior_resized), dim=1))

        fused = self.dropout(fused)
        return self.linear_pred(fused)
