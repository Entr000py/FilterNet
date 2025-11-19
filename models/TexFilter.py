import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from models.temporal_conv import TemporalConv


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block to recalibrate temporal features."""

    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, n, _ = x.size()
        # Only squeeze the temporal dimension to avoid relying on None-sized pools
        y = x.mean(dim=-1, keepdim=True).view(b, d, n)
        y = y.permute(0, 2, 1).reshape(b * n, d)
        y = self.fc(y).view(b, n, d, 1).permute(0, 2, 1, 3)
        return x * y.expand_as(x)


class TemporalConvEmbedding(nn.Module):
    def __init__(self, seq_len: int, embed_size: int, dilation_factor: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.temporal_conv = TemporalConv(
            cin=1, cout=embed_size, dilation_factor=dilation_factor, seq_len=seq_len
        )
        self.se_block = SEBlock(channel=embed_size, reduction=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, L]
        x_tc = x.unsqueeze(1)
        x_tc = self.temporal_conv(x_tc)
        x_tc = self.se_block(x_tc)
        x_tc = x_tc.mean(dim=-1)
        x_tc = x_tc.permute(0, 2, 1).contiguous()
        return x_tc


class EnergyGatedTexFilter(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.embed_size = configs.embed_size
        self.scale = 0.02
        self.threshold_param = nn.Parameter(torch.tensor(0.5))
        self.sparsity_threshold = 0.01
        self.n_freq = configs.enc_in // 2 + 1

        self.global_weight = nn.Parameter(
            torch.randn(self.n_freq, self.embed_size, 2) * 0.02
        )

        self.w = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

    def create_adaptive_energy_mask(self, x_fft: torch.Tensor) -> torch.Tensor:
        # Sum over embedding dimension to match the mask design in TSLANet
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        b = energy.size(0)
        flat_energy = energy.view(b, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(b, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        # Straight-through estimator: forward uses the hard mask, backward keeps gradients
        hard_mask = (normalized_energy > self.threshold_param).float()
        adaptive_mask = (hard_mask - normalized_energy).detach() + normalized_energy
        return adaptive_mask.unsqueeze(-1)

    def generate_dynamic_weight(self, x: torch.Tensor) -> torch.Tensor:
        o1_real = F.relu(
            torch.einsum("bid,d->bid", x.real, self.w[0])
            - torch.einsum("bid,d->bid", x.imag, self.w[1])
            + self.rb1
        )
        o1_imag = F.relu(
            torch.einsum("bid,d->bid", x.imag, self.w[0])
            + torch.einsum("bid,d->bid", x.real, self.w[1])
            + self.ib1
        )
        o2_real = (
            torch.einsum("bid,d->bid", o1_real, self.w1[0])
            - torch.einsum("bid,d->bid", o1_imag, self.w1[1])
            + self.rb2
        )
        o2_imag = (
            torch.einsum("bid,d->bid", o1_imag, self.w1[0])
            + torch.einsum("bid,d->bid", o1_real, self.w1[1])
            + self.ib2
        )
        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        return torch.view_as_complex(y)

    def forward(self, x_fft: torch.Tensor) -> torch.Tensor:
        global_weight = self.global_weight
        if global_weight.shape[0] != x_fft.shape[1]:
            weight = global_weight.permute(1, 2, 0).reshape(1, -1, global_weight.shape[0])
            weight = F.interpolate(
                weight, size=x_fft.shape[1], mode="linear", align_corners=False
            )
            global_weight = weight.view(self.embed_size, 2, -1).permute(2, 0, 1).contiguous()
        global_w = torch.view_as_complex(global_weight)
        out_global = x_fft * global_w

        energy_mask = self.create_adaptive_energy_mask(x_fft)
        dynamic_w = self.generate_dynamic_weight(x_fft)
        out_dynamic = x_fft * dynamic_w * energy_mask.to(x_fft.device)

        return out_global + out_dynamic


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.dropout_rate = configs.dropout

        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.embedding = nn.Linear(self.seq_len, self.embed_size)
        self.temporal_embed = TemporalConvEmbedding(
            seq_len=self.seq_len,
            embed_size=self.embed_size,
            dilation_factor=getattr(configs, "temporal_conv_dilation", 1),
        )
        self.embed_gate = nn.Sequential(
            nn.Linear(self.embed_size * 2, self.embed_size),
            nn.Sigmoid(),
        )

        self.tex_filter = EnergyGatedTexFilter(configs)

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )

        self.output = nn.Linear(self.embed_size, self.pred_len)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.layernorm1 = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [Batch, Input length, Channel]
        B, L, N = x.shape
        x = self.revin_layer(x, 'norm')

        x_perm = x.permute(0, 2, 1)
        x_lin = self.embedding(x_perm)
        x_conv = self.temporal_embed(x_perm)
        gate = self.embed_gate(torch.cat([x_lin, x_conv], dim=-1))
        x = x_lin + gate * x_conv
        x = self.layernorm(x)

        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        x_fft_filtered = self.tex_filter(x_fft)
        x = torch.fft.irfft(x_fft_filtered, n=N, dim=1, norm="ortho")

        x = self.layernorm1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)

        x = self.revin_layer(x, 'denorm')

        return x
