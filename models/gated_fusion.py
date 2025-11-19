import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexMSTFusion(nn.Module):
    """
    复数特征的多尺度门控融合（MSTF）。

    输入为若干路形状 [B, F, D] 的复数特征，先将实部和虚部在最后一维拼接为
    实值特征 [B, F, 2D]，再参照 6_3_MSTF.py 的思路在“尺度维”上做 softmax 加权融合，
    最后再拆分回实部和虚部，输出融合后的复数特征。
    """

    def __init__(self, channels: int):
        """
        参数
        ----
        channels: 复数特征的嵌入维 D（而非 2D）。
        """
        super().__init__()
        self.channels = channels
        out_channels = channels * 2  # 实部 + 虚部

        self.project1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.project2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """
        参数
        ----
        inputs: List[Tensor]，每个张量形状为 [B, F, D] 的复数张量。

        返回
        ----
        形状为 [B, F, D] 的复数张量，为多路频域特征的 MSTF 融合结果。
        """
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("ComplexMSTFusion expects a list/tuple of tensors.")
        if len(inputs) < 2:
            raise ValueError("ComplexMSTFusion expects at least two inputs to fuse.")

        x0 = inputs[0]
        if not torch.is_complex(x0):
            raise TypeError("ComplexMSTFusion expects complex-valued tensors.")

        b, f, d = x0.shape
        m = len(inputs)

        # 将每一路复数特征转换为实值特征 [B, F, 2D]，再补上 N=1 维：[B, F, 1, 2D]
        real_inputs = []
        for x in inputs:
            if x.shape != x0.shape:
                raise ValueError("All inputs to ComplexMSTFusion must share the same shape.")
            if not torch.is_complex(x):
                raise TypeError("ComplexMSTFusion expects complex-valued tensors.")
            x_ri = torch.cat([x.real, x.imag], dim=-1).unsqueeze(2)
            real_inputs.append(x_ri)  # [B, F, 1, 2D]

        # 多尺度拼接到 batch 维，再 reshape 出尺度维 M
        x_cat = torch.cat(real_inputs, dim=0)  # [M*B, F, 1, 2D]
        x_cat = x_cat.reshape(m, f, b, -1)  # [M, F, B, 2D]

        # project1：在通道维 2D 上做 1x1 卷积，学习尺度间相关性
        x_for_conv = x_cat.permute(0, 3, 2, 1)  # [M, 2D, B, F]
        x_for_conv = self.project1(x_for_conv)
        x_weight_feat = x_for_conv.permute(0, 3, 2, 1)  # [M, F, B, 2D]

        # 在尺度维 M 上 softmax，得到每一路的自适应权重
        weight = F.softmax(x_weight_feat, dim=0)  # [M, F, B, 2D]

        # 对原始实值特征做加权和
        x_value = x_cat  # [M, F, B, 2D]
        fused = (weight * x_value).sum(dim=0)  # [F, B, 2D]

        # reshape 回 [B, F, 1, 2D] 再做 project2
        fused = fused.permute(1, 0, 2).unsqueeze(2)  # [B, F, 1, 2D]
        fused = fused.permute(0, 3, 2, 1)  # [B, 2D, 1, F]
        fused = self.project2(fused)
        fused = fused.permute(0, 3, 2, 1)  # [B, F, 1, 2D]
        fused = fused.squeeze(2)  # [B, F, 2D]

        # 拆分为实部和虚部，重组为复数张量
        real, imag = fused.split(self.channels, dim=-1)
        return torch.complex(real, imag)

