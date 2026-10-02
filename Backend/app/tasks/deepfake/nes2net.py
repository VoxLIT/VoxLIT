"""Nes2Net-X (Model F) - XLS-R front end + Nested Res2Net TDNN back end.

Liu, Truong, Das, Lee & Li, "Nes2Net: A Lightweight Nested Architecture for
Foundation Model Driven Speech Anti-Spoofing", IEEE TIFS 2025
(arXiv:2504.05657). Reference code: Liu-Tianchi/Nes2Net_ASVspoof_ITW
(model_scripts/wav2vec2_Nes2Net_X.py), MIT. Checkpoint: the authors' single
Nes2Net-X model (nes2net_x_DF1.65.pth, trained on ASVspoof 2019 LA with
RawBoost), re-hosted unchanged at SpeechAntiSpoofingBenchmarks/Nes2Net.
Released configuration: Nes_ratio [8, 8], SE_ratio [1], dilation 2, mean
pooling (the repo's defaults; the Arena's meta.yaml records the same).

The back end is only ~0.5M parameters: XLS-R's 1024 channels are split into
8 groups of 128, and each group is passed through a Res2Net block that is
itself split 8 ways -- hence "nested". QUIRK: inside each block the
sub-groups are stacked along a NEW trailing axis and blended by a learned
`weighted_sum`, rather than added as in a plain Res2Net. Transcribed exactly.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from . import ssl_frontend

REPO_FILENAME = "nes2net_x_DF1.65.pth"
SSL_DIM = 1024
NES_RATIO = (8, 8)
SE_RATIO = 1
DILATION = 2


class SEModule(nn.Module):
    def __init__(self, channels: int, se_ratio: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Bottle2neck(nn.Module):
    def __init__(self, planes: int, kernel_size: int, dilation: int, scale: int, se_ratio: int) -> None:
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(planes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        pad = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList(
            nn.Conv2d(width, width, kernel_size=(kernel_size, 1), dilation=(dilation, 1), padding=(pad, 0))
            for _ in range(self.nums)
        )
        self.bns = nn.ModuleList(nn.BatchNorm2d(width) for _ in range(self.nums))
        self.weighted_sum = nn.ParameterList(
            nn.Parameter(torch.ones(1, 1, 1, i + 2) / (i + 2)) for i in range(self.nums)
        )
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes, se_ratio)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x))).unsqueeze(-1)  # (B, C, T, 1)

        spx = torch.split(out, self.width, 1)
        sp = spx[self.nums]
        pieces = []
        for i in range(self.nums):
            sp = torch.cat((sp, spx[i]), -1)
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            pieces.append(torch.sum(sp * self.weighted_sum[i], dim=-1))
        pieces.append(spx[self.nums].squeeze(-1))
        out = torch.cat(pieces, 1)

        out = self.bn3(self.relu(self.conv3(out)))
        return self.se(out) + residual


class NestedRes2NetTDNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        outer, inner = NES_RATIO
        self.nes_ratio = outer
        self.C = SSL_DIM // outer
        self.Build_in_Res2Nets = nn.ModuleList(
            Bottle2neck(self.C, kernel_size=3, dilation=DILATION, scale=inner, se_ratio=SE_RATIO)
            for _ in range(outer - 1)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.C) for _ in range(outer - 1))
        self.bn = nn.BatchNorm1d(SSL_DIM)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(SSL_DIM, 2)

    def forward(self, x):
        spx = torch.split(x, self.C, 1)
        pieces = []
        sp = None
        for i in range(self.nes_ratio - 1):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](self.relu(self.Build_in_Res2Nets[i](sp)))
            pieces.append(sp)
        pieces.append(spx[-1])
        out = self.relu(self.bn(torch.cat(pieces, 1)))
        return self.fc(torch.mean(out, dim=-1))


class Nes2NetX(nn.Module):
    def __init__(self, ssl_model) -> None:
        super().__init__()
        self.ssl_model = ssl_model
        self.Nested_Res2Net_TDNN = NestedRes2NetTDNN()

    def forward(self, waveform):
        features = self.ssl_model(waveform).last_hidden_state.permute(0, 2, 1)
        return self.Nested_Res2Net_TDNN(features)

    @property
    def classifier(self):
        return self.Nested_Res2Net_TDNN.fc


def load(repo_id: str, revision: str = "main", token: str | None = None) -> Nes2NetX:
    checkpoint = ssl_frontend.download_state_dict(repo_id, REPO_FILENAME, revision, token)
    model = Nes2NetX(ssl_frontend.new_ssl_model())
    ssl_frontend.load_into(model, checkpoint)
    return model
