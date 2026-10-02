"""Wav2Vec2-AASIST (Model D) - XLS-R front end + AASIST graph-attention back end.

Tak, Todisco, Wang, Jung, Yamagishi & Evans, "Automatic speaker verification
spoofing and deepfake detection using wav2vec 2.0 and data augmentation",
Odyssey 2022 (arXiv:2202.12233). Reference code: TakHemlata/SSL_Anti-spoofing
(model.py), MIT. Checkpoint: the authors' LA_model.pth (trained on ASVspoof
2019 LA with RawBoost), as re-hosted unchanged by the Speech DF Arena
maintainers at SpeechAntiSpoofingBenchmarks/W2V2-AASIST.

Tier B for the same reason as Model C: the checkpoint is a bare state_dict,
so the architecture below is transcribed from the reference model.py and
loaded with strict=True. The behaviour that would otherwise fail SILENTLY is
marked QUIRK.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import ssl_frontend

REPO_FILENAME = "LA_model.pth"
SSL_DIM = 1024

# AASIST hyper-parameters, fixed in the reference Model.__init__.
FILTS = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
GAT_DIMS = [64, 32]
POOL_RATIOS = [0.5, 0.5, 0.5, 0.5]
TEMPERATURES = [2.0, 2.0, 100.0, 100.0]
# 128 projected channels max-pooled by 3 -> 42 spectral nodes. Only true for
# the 64600-sample window, which is why that window is mandatory.
SPECTRAL_NODES = 42


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.zeros(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x):
        x = self.input_drop(x)
        att_map = _pairwise(x)
        att_map = torch.matmul(torch.tanh(self.att_proj(att_map)), self.att_weight)
        att_map = F.softmax(att_map / self.temp, dim=-2)
        x = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)
        return self.act(_batch_norm_nodes(self.bn, x))


class HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous stacking graph attention (spectral + temporal + master)."""

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = nn.Parameter(torch.zeros(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.zeros(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.zeros(out_dim, 1))
        self.att_weightM = nn.Parameter(torch.zeros(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x1, x2, master=None):
        num_type1, num_type2 = x1.size(1), x2.size(1)
        x = torch.cat([self.proj_type1(x1), self.proj_type2(x2)], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)

        att_map = self._derive_att_map(x, num_type1)
        master = self._update_master(x, master)

        x = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)
        x = self.act(_batch_norm_nodes(self.bn, x))
        return x.narrow(1, 0, num_type1), x.narrow(1, num_type1, num_type2), master

    def _derive_att_map(self, x, num_type1: int):
        att_map = torch.tanh(self.att_proj(_pairwise(x)))
        board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        n = num_type1
        board[:, :n, :n, :] = torch.matmul(att_map[:, :n, :n, :], self.att_weight11)
        board[:, n:, n:, :] = torch.matmul(att_map[:, n:, n:, :], self.att_weight22)
        # QUIRK: both off-diagonal blocks use att_weight12 (there is no
        # att_weight21 in the reference or the checkpoint).
        board[:, :n, n:, :] = torch.matmul(att_map[:, :n, n:, :], self.att_weight12)
        board[:, n:, :n, :] = torch.matmul(att_map[:, n:, :n, :], self.att_weight12)
        return F.softmax(board / self.temp, dim=-2)

    def _update_master(self, x, master):
        att_map = torch.matmul(torch.tanh(self.att_projM(x * master)), self.att_weightM)
        att_map = F.softmax(att_map / self.temp, dim=-2)
        pooled = torch.matmul(att_map.squeeze(-1).unsqueeze(1), x)
        return self.proj_with_attM(pooled) + self.proj_without_attM(master)


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: float) -> None:
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        scores = self.sigmoid(self.proj(self.drop(h)))
        _, n_nodes, n_feat = h.size()
        keep = max(int(n_nodes * self.k), 1)
        _, idx = torch.topk(scores, keep, dim=1)
        return torch.gather(h * scores, 1, idx.expand(-1, -1, n_feat))


class ResidualBlock(nn.Module):
    def __init__(self, nb_filts, first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1), stride=1)
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(
                nb_filts[0], nb_filts[1], padding=(0, 1), kernel_size=(1, 3), stride=1
            )

    def forward(self, x):
        identity = x
        # QUIRK: the reference computes bn1/selu for non-first blocks and then
        # DISCARDS it -- conv1 is applied to the raw input `x`. The trained
        # weights expect that, so it is reproduced (bn1 still has to exist
        # for the checkpoint to load strictly). Do not "fix" it.
        out = self.conv1(x)
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return out + identity


class W2V2AASIST(nn.Module):
    def __init__(self, ssl_model) -> None:
        super().__init__()
        self.ssl_model = ssl_model
        self.LL = nn.Linear(SSL_DIM, FILTS[0])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(ResidualBlock(FILTS[1], first=True)),
            nn.Sequential(ResidualBlock(FILTS[2])),
            nn.Sequential(ResidualBlock(FILTS[3])),
            nn.Sequential(ResidualBlock(FILTS[4])),
            nn.Sequential(ResidualBlock(FILTS[4])),
            nn.Sequential(ResidualBlock(FILTS[4])),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        self.pos_S = nn.Parameter(torch.zeros(1, SPECTRAL_NODES, FILTS[-1][-1]))
        self.master1 = nn.Parameter(torch.zeros(1, 1, GAT_DIMS[0]))
        self.master2 = nn.Parameter(torch.zeros(1, 1, GAT_DIMS[0]))

        self.GAT_layer_S = GraphAttentionLayer(FILTS[-1][-1], GAT_DIMS[0], TEMPERATURES[0])
        self.GAT_layer_T = GraphAttentionLayer(FILTS[-1][-1], GAT_DIMS[0], TEMPERATURES[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(GAT_DIMS[0], GAT_DIMS[1], TEMPERATURES[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(GAT_DIMS[1], GAT_DIMS[1], TEMPERATURES[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(GAT_DIMS[0], GAT_DIMS[1], TEMPERATURES[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(GAT_DIMS[1], GAT_DIMS[1], TEMPERATURES[2])

        self.pool_S = GraphPool(POOL_RATIOS[0], GAT_DIMS[0], 0.3)
        self.pool_T = GraphPool(POOL_RATIOS[1], GAT_DIMS[0], 0.3)
        self.pool_hS1 = GraphPool(POOL_RATIOS[2], GAT_DIMS[1], 0.3)
        self.pool_hT1 = GraphPool(POOL_RATIOS[2], GAT_DIMS[1], 0.3)
        self.pool_hS2 = GraphPool(POOL_RATIOS[2], GAT_DIMS[1], 0.3)
        self.pool_hT2 = GraphPool(POOL_RATIOS[2], GAT_DIMS[1], 0.3)

        self.out_layer = nn.Linear(5 * GAT_DIMS[1], 2)

    def forward(self, waveform):
        x = self.LL(self.ssl_model(waveform).last_hidden_state)
        x = x.transpose(1, 2).unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.selu(self.first_bn(x))

        x = self.encoder(x)
        x = self.selu(self.first_bn1(x))
        w = self.attention(x)

        # Spectral graph: attend over time, one node per channel band.
        m = torch.sum(x * F.softmax(w, dim=-1), dim=-1)
        out_S = self.pool_S(self.GAT_layer_S(m.transpose(1, 2) + self.pos_S))

        # Temporal graph: attend over channels, one node per frame group.
        m1 = torch.sum(x * F.softmax(w, dim=-2), dim=-2)
        out_T = self.pool_T(self.GAT_layer_T(m1.transpose(1, 2)))

        # QUIRK: the reference passes the UNexpanded self.master1/2 here (it
        # builds expanded copies and never uses them). Identical at batch 1.
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1, out_T1 = self.pool_hS1(out_S1), self.pool_hT1(out_T1)
        aug_T, aug_S, aug_M = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1, out_S1, master1 = out_T1 + aug_T, out_S1 + aug_S, master1 + aug_M

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2, out_T2 = self.pool_hS2(out_S2), self.pool_hT2(out_T2)
        aug_T, aug_S, aug_M = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2, out_S2, master2 = out_T2 + aug_T, out_S2 + aug_S, master2 + aug_M

        out_T = torch.max(self.drop_way(out_T1), self.drop_way(out_T2))
        out_S = torch.max(self.drop_way(out_S1), self.drop_way(out_S2))
        master = torch.max(self.drop_way(master1), self.drop_way(master2))

        last_hidden = torch.cat(
            [
                torch.max(torch.abs(out_T), dim=1)[0],
                torch.mean(out_T, dim=1),
                torch.max(torch.abs(out_S), dim=1)[0],
                torch.mean(out_S, dim=1),
                master.squeeze(1),
            ],
            dim=1,
        )
        return self.out_layer(self.drop(last_hidden))

    @property
    def classifier(self):
        """The layer the decision is read from (embedding view hooks it)."""
        return self.out_layer


def _pairwise(x):
    nodes = x.size(1)
    x = x.unsqueeze(2).expand(-1, -1, nodes, -1)
    return x * x.transpose(1, 2)


def _batch_norm_nodes(bn, x):
    size = x.size()
    return bn(x.view(-1, size[-1])).view(size)


def load(repo_id: str, revision: str = "main", token: str | None = None) -> W2V2AASIST:
    checkpoint = ssl_frontend.download_state_dict(repo_id, REPO_FILENAME, revision, token)
    model = W2V2AASIST(ssl_frontend.new_ssl_model())
    ssl_frontend.load_into(model, checkpoint)
    return model
