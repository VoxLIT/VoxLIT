"""XLSR-SLS (Model E) - XLS-R front end + Sensitive Layer Selection back end.

Zhang, Wen & Hu, "Audio Deepfake Detection with Self-Supervised XLS-R and SLS
Classifier", ACM Multimedia 2024 (doi:10.1145/3664647.3681345). Reference
code: QiShanZhang/SLSforASVspoof-2021-DF (model.py), MIT. Checkpoint: the
paper's MMpaper_model.pth (trained on ASVspoof 2019 LA), re-hosted unchanged
at SpeechAntiSpoofingBenchmarks/XLSR-SLS. Best open-source system in the
Speech DF Arena (Dowerah et al., 2025, Table 3: 13.84% average EER, 15.68%
pooled EER over 14 evaluation sets).

Instead of reading only XLS-R's last layer, SLS pools all 24 transformer
layers, gates each with a learned sigmoid weight, and sums them. QUIRKS:

* The layers it reads are fairseq's `layer_results` from the authors'
  PATCHED fairseq (shipped as a zip in their repo, which flips
  `if tgt_layer is not None` to `is None` so the list is filled). Each entry
  is a layer's raw output, before the encoder's final LayerNorm -- see
  ssl_frontend.layer_outputs for why transformers' hidden_states differ.
* fc1 is 22847 wide = (201 frames / 3) x (1024 / 3) after a 3x3 max-pool,
  so the 64600-sample window is structural, not a preference.
* The head ends in SELU then LogSoftmax. Softmax over log-probabilities
  gives back the same probabilities, so the shared scoring path is valid.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import ssl_frontend

REPO_FILENAME = "MMpaper_model.pth"
SSL_DIM = 1024
FLAT_DIM = 22_847


class XLSRSLS(nn.Module):
    def __init__(self, ssl_model) -> None:
        super().__init__()
        self.ssl_model = ssl_model
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(SSL_DIM, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(FLAT_DIM, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, waveform):
        layers = ssl_frontend.layer_outputs(self.ssl_model, waveform)

        # (batch, layers, frames, dim) and one pooled descriptor per layer.
        stack = torch.stack(layers, dim=1)
        pooled = stack.mean(dim=2)

        gates = self.sig(self.fc0(pooled)).unsqueeze(-1)  # (batch, layers, 1, 1)
        fused = torch.sum(stack * gates, dim=1).unsqueeze(1)

        x = self.selu(self.first_bn(fused))
        x = torch.flatten(F.max_pool2d(x, (3, 3)), 1)
        x = self.selu(self.fc1(x))
        x = self.selu(self.fc3(x))
        return self.logsoftmax(x)

    @property
    def classifier(self):
        return self.fc3


def load(repo_id: str, revision: str = "main", token: str | None = None) -> XLSRSLS:
    checkpoint = ssl_frontend.download_state_dict(repo_id, REPO_FILENAME, revision, token)
    model = XLSRSLS(ssl_frontend.new_ssl_model())
    ssl_frontend.load_into(model, checkpoint)
    return model
