import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from models.temporal_conv import TemporalConv


class SEBlock(nn.Module):
    """通道注意力（SE）模块，用于重标定时间特征。

    本模块将最后一维视为时间维，在时间维上做压缩（平均池化），
    对嵌入维 D 做逐通道的加权，从而实现通道级别的注意力。

    约定张量形状
    -----------
    x:   [B, D, N, T]  其中 B: batch, D: embedding 维度, N: 变量/节点数, T: 时间长度
    out: 与 x 相同形状 [B, D, N, T]
    """

    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """在时间维上聚合并进行通道加权。

        参数
        ----
        x: 形状为 [B, D, N, T] 的张量。

        返回
        ----
        与输入形状一致的张量 [B, D, N, T]，但在通道维 D 上进行了重标定。
        """
        b, d, n, _ = x.size()
        # 只在时间维 T 上做平均池化，避免依赖动态池化算子
        y = x.mean(dim=-1, keepdim=True).view(b, d, n)
        y = y.permute(0, 2, 1).reshape(b * n, d)
        y = self.fc(y).view(b, n, d, 1).permute(0, 2, 1, 3)
        return x * y.expand_as(x)


class TemporalConvEmbedding(nn.Module):
    def __init__(self, seq_len: int, embed_size: int, dilation_factor: int = 1):
        """基于时间卷积的嵌入模块。

        在时间维上使用空洞卷积进行特征提取，然后在时间维上聚合，
        得到每个变量的定长嵌入。

        参数
        ----
        seq_len: 输入序列长度 L。
        embed_size: 输出嵌入维度 D。
        dilation_factor: 时间卷积中的扩张系数。
        """
        super().__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.temporal_conv = TemporalConv(
            cin=1, cout=embed_size, dilation_factor=dilation_factor, seq_len=seq_len
        )
        self.se_block = SEBlock(channel=embed_size, reduction=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：生成时间卷积嵌入。

        参数
        ----
        x: 形状为 [B, N, L] 的张量，
           B 为 batch 大小，N 为变量/通道数，L 为输入时间长度。

        返回
        ----
        形状为 [B, N, D] 的张量，其中 D == embed_size。
        """
        # x: [B, N, L]
        x_tc = x.unsqueeze(1)
        x_tc = self.temporal_conv(x_tc)
        x_tc = self.se_block(x_tc)
        x_tc = x_tc.mean(dim=-1)
        x_tc = x_tc.permute(0, 2, 1).contiguous()
        return x_tc


class EnergyGatedTexFilter(nn.Module):
    def __init__(self, configs):
        """频域纹理滤波模块，带可学习门控。

        本模块在“通道维”的 FFT 频域上操作复数特征，
        同时学习一个全局滤波器和一个与输入相关的动态滤波器，
        并由局部能量进行门控以实现自适应抑制/增强。

        参数
        ----
        configs: 配置对象，至少包含
            - enc_in: 输入通道数 C。
            - embed_size: 嵌入维度 D。
        """
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
        """构建基于能量的可微掩码。

        参数
        ----
        x_fft: 形状为 [B, F, D] 的复数张量，
               F 为在通道维上做 FFT 后的频点数，D 为嵌入维。

        返回
        ----
        形状为 [B, F, 1] 的实数掩码，前向为硬门控，反向使用直通估计。
        """
        # 在嵌入维 D 上累加能量，使设计与 TSLANet 掩码保持一致
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        b = energy.size(0)
        flat_energy = energy.view(b, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(b, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        # 直通估计：前向用硬掩码，反向保留梯度
        hard_mask = (normalized_energy > self.threshold_param).float()
        adaptive_mask = (hard_mask - normalized_energy).detach() + normalized_energy
        return adaptive_mask.unsqueeze(-1)

    def generate_dynamic_weight(self, x: torch.Tensor) -> torch.Tensor:
        """生成与输入相关的动态复数频域滤波器。

        参数
        ----
        x: 形状为 [B, F, D] 的复数张量。

        返回
        ----
        形状为 [B, F, D] 的复数张量，表示每个样本的动态频域权重。
        """
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
        # 先在最后一维堆叠实部和虚部，再通过 torch.complex 显式构造复数，
        # 避免 torch.view_as_complex 对底层 storage_offset 的严格要求。
        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        real, imag = y.unbind(dim=-1)
        return torch.complex(real, imag)

    def forward(self, x_fft: torch.Tensor) -> torch.Tensor:
        """在频域上同时应用全局与动态复数滤波器。

        参数
        ----
        x_fft: 沿通道维做 FFT 得到的复数张量，形状为 [B, F, D]。

        返回
        ----
        形状为 [B, F, D] 的复数张量，为滤波后的频域特征。
        """
        global_weight = self.global_weight
        if global_weight.shape[0] != x_fft.shape[1]:
            weight = global_weight.permute(1, 2, 0).reshape(1, -1, global_weight.shape[0])
            weight = F.interpolate(
                weight, size=x_fft.shape[1], mode="linear", align_corners=False
            )
            global_weight = weight.view(self.embed_size, 2, -1).permute(2, 0, 1).contiguous()
        # 使用 torch.complex 明确构造复数权重，避免 view_as_complex 的 storage_offset 限制
        real_w, imag_w = global_weight.unbind(dim=-1)
        global_w = torch.complex(real_w, imag_w)
        out_global = x_fft * global_w

        energy_mask = self.create_adaptive_energy_mask(x_fft)
        dynamic_w = self.generate_dynamic_weight(x_fft)
        out_dynamic = x_fft * dynamic_w * energy_mask.to(x_fft.device)

        return out_global + out_dynamic


class Model(nn.Module):

    def __init__(self, configs):
        """主预测模型，内置频域纹理滤波器。

        参数
        ----
        configs: 配置对象，至少包含
            - seq_len: 输入序列长度 L。
            - pred_len: 预测长度 P。
            - enc_in: 输入通道数 C。
            - embed_size: 嵌入维度 D。
            - hidden_size: MLP 头部的隐藏维度 H。
            - dropout: dropout 比例。
        """
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
        """前向传播。

        模型先对输入做可逆归一化，然后为每个通道构建时间卷积嵌入，
        在“通道维”的频域上应用可学习滤波器，最后映射到预测窗口长度。

        参数
        ----
        x: 形状为 [B, L, C] 的输入序列。
        x_mark_enc: 预留的时间特征（当前未使用，保持接口兼容）。
        x_dec: 解码器输入（当前未使用）。
        x_mark_dec: 解码器时间特征（当前未使用，保持接口兼容）。
        mask: 可选掩码（当前未使用）。

        返回
        ----
        形状为 [B, P, C] 的预测结果，其中 P == pred_len。
        """
        # x: [B, L, C]  (batch, 输入长度, 通道数)
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
