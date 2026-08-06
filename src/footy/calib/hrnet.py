"""Clean-room HRNetV2-W48 keypoint/line heatmap backbone.

The architecture here was reconstructed solely from the tensor shapes and key
names of the released checkpoints (``SV_kp.pth``, ``SV_lines.pth``); no GPL
source code was consulted or copied. The weights themselves are CC-BY-4.0 from
Zenodo record 12626395.

Topology (verified key-by-key against the checkpoints):

- Stem: two 3x3 stride-2 convs (3->64->64), each followed by BN + ReLU.
- ``layer1``: four torchvision-style Bottleneck blocks (64->256).
- Four parallel branches at widths 48/96/192/384, built up through
  ``transition1``..``transition3`` and fused in ``stage2`` (1 module),
  ``stage3`` (4 modules) and ``stage4`` (3 modules); every module keeps
  multi-scale output (fuse layers exist for all branch pairs).
- Head input is 784 channels: the 720-channel concat of the four upsampled
  branches plus the 64-channel stem feature (post ``bn2``/ReLU, pre
  ``layer1``), which is the only 64-channel tensor at branch-0 resolution.
  ``stem_position`` selects whether the stem feature is concatenated first
  (default) or last.
- Head: 1x1 conv 784->784 + BN + ReLU + 1x1 conv 784->``n_out``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_WIDTHS = (48, 96, 192, 384)
_STEM_CHANNELS = 64


def _conv_bn(c_in: int, c_out: int, kernel: int, stride: int) -> nn.Sequential:
    """3x3/1x1 conv (no bias) followed by BatchNorm, matching checkpoint key layout."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(c_out),
    )


class Bottleneck(nn.Module):
    """Torchvision-style bottleneck residual block (expansion 4)."""

    expansion = 4

    def __init__(self, c_in: int, planes: int, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class BasicBlock(nn.Module):
    """Two 3x3 convs with a residual connection (expansion 1, stride 1)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class HRModule(nn.Module):
    """Parallel per-branch BasicBlocks followed by full cross-resolution fusion."""

    def __init__(self, widths: tuple[int, ...], blocks_per_branch: int = 4) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(*[BasicBlock(w) for _ in range(blocks_per_branch)]) for w in widths
        )
        fuse_layers: list[nn.ModuleList] = []
        for i, w_out in enumerate(widths):
            row: list[nn.Module | None] = []
            for j, w_in in enumerate(widths):
                if j == i:
                    row.append(None)
                elif j > i:
                    # Coarser -> finer: 1x1 conv + BN; upsampled to size in forward().
                    row.append(_conv_bn(w_in, w_out, 1, 1))
                else:
                    # Finer -> coarser: (i - j) strided 3x3 conv blocks; intermediate
                    # steps keep the source width, the last projects to w_out.
                    steps: list[nn.Module] = []
                    for k in range(i - j):
                        last = k == i - j - 1
                        step = _conv_bn(w_in, w_out if last else w_in, 3, 2)
                        if not last:
                            step.append(nn.ReLU(inplace=True))
                        steps.append(step)
                    row.append(nn.Sequential(*steps))
            fuse_layers.append(nn.ModuleList(row))
        self.fuse_layers = nn.ModuleList(fuse_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs = [branch(x) for branch, x in zip(self.branches, xs, strict=True)]
        outs = []
        for i, row in enumerate(self.fuse_layers):
            acc = xs[i]
            for j, fuse in enumerate(row):
                if fuse is None:
                    continue
                y = fuse(xs[j])
                if j > i:
                    y = F.interpolate(y, size=xs[i].shape[-2:], mode="nearest")
                acc = acc + y
            outs.append(self.relu(acc))
        return outs


class HRNetV2(nn.Module):
    """HRNetV2-W48 producing ``n_out`` heatmaps at 1/4 input resolution."""

    def __init__(self, n_out: int = 58, stem_position: str = "last") -> None:
        super().__init__()
        if stem_position not in ("first", "last"):
            raise ValueError(f"stem_position must be 'first' or 'last', got {stem_position!r}")
        self.stem_position = stem_position

        self.conv1 = nn.Conv2d(3, _STEM_CHANNELS, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(_STEM_CHANNELS)
        self.conv2 = nn.Conv2d(_STEM_CHANNELS, _STEM_CHANNELS, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(_STEM_CHANNELS)
        self.relu = nn.ReLU(inplace=True)

        downsample = _conv_bn(_STEM_CHANNELS, 256, 1, 1)
        self.layer1 = nn.Sequential(
            Bottleneck(_STEM_CHANNELS, 64, downsample),
            *[Bottleneck(256, 64) for _ in range(3)],
        )

        # transition1 splits layer1's 256 channels into the first two branches.
        self.transition1 = nn.ModuleList(
            [
                nn.Sequential(*_conv_bn(256, _WIDTHS[0], 3, 1), nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Sequential(*_conv_bn(256, _WIDTHS[1], 3, 2), nn.ReLU(inplace=True))
                ),
            ]
        )
        self.stage2 = nn.Sequential(HRModule(_WIDTHS[:2]))

        # transition2/3 each add one branch by striding down the previous finest.
        self.transition2 = self._new_branch_transition(2)
        self.stage3 = nn.Sequential(*[HRModule(_WIDTHS[:3]) for _ in range(4)])
        self.transition3 = self._new_branch_transition(3)
        self.stage4 = nn.Sequential(*[HRModule(_WIDTHS) for _ in range(3)])

        head_in = _STEM_CHANNELS + sum(_WIDTHS)  # 64 + 720 = 784
        self.head = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(head_in, head_in, 1),
                nn.BatchNorm2d(head_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_in, n_out, 1),
            )
        )

    @staticmethod
    def _new_branch_transition(index: int) -> nn.ModuleList:
        """Pass-through for existing branches, strided conv creating branch ``index``."""
        layers: list[nn.Module | None] = [None] * index
        layers.append(
            nn.Sequential(
                nn.Sequential(
                    *_conv_bn(_WIDTHS[index - 1], _WIDTHS[index], 3, 2), nn.ReLU(inplace=True)
                )
            )
        )
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        stem = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(stem)

        xs = [self.transition1[0](x), self.transition1[1](x)]
        xs = self.stage2(xs)
        xs = [*xs, self.transition2[2](xs[-1])]
        xs = self.stage3(xs)
        xs = [*xs, self.transition3[3](xs[-1])]
        xs = self.stage4(xs)

        size = xs[0].shape[-2:]
        ups = [xs[0]] + [F.interpolate(b, size=size, mode="nearest") for b in xs[1:]]
        if self.stem_position == "first":
            feats = torch.cat([stem, *ups], dim=1)
        else:
            feats = torch.cat([*ups, stem], dim=1)
        return self.head(feats)


def load_checkpoint(
    path: str | Path, n_out: int | None = None, stem_position: str = "last"
) -> HRNetV2:
    """Build an :class:`HRNetV2` and load ``path`` strictly, inferring ``n_out`` if None."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if n_out is None:
        n_out = state["head.0.3.weight"].shape[0]
    model = HRNetV2(n_out=n_out, stem_position=stem_position)
    model.load_state_dict(state, strict=True)
    return model
