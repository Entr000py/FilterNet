import torch
import torch.nn as nn

from layers.RevIN import RevIN
from models.temporal_conv import temporal_conv

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # 输入序列长度
        self.pred_len = configs.pred_len  # 输出预测序列长度
        self.scale = 0.02  # 卷积权重的初始化尺度
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)  # 归一化前处理

        self.embed_size = self.seq_len  # 嵌入向量维度等于序列长度
        self.hidden_size = configs.hidden_size  # 全连接层内部维度
        
        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))  # 频域卷积核

        temporal_channels = max(1, getattr(configs, "temporal_conv_channels", 4))
        temporal_dilation = max(1, getattr(configs, "temporal_conv_dilation", 1))
        self.temporal_conv = temporal_conv(
            cin=1,
            cout=temporal_channels,
            dilation_factor=temporal_dilation,
            seq_len=self.seq_len,
        )

        # 简单的前向全连接结构：嵌入 -> 非线性 -> 输出
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )


    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x):
        z = x  # x: (B, seq_len, enc_in)
        z = self.revin_layer(z, 'norm')  # normalized, shape unchanged
        x = z

        x = x.permute(0, 2, 1)  # -> (B, enc_in, seq_len)
        x = x.unsqueeze(1)  # -> (B, 1, enc_in, seq_len)

        x = self.temporal_conv(x)  # -> (B, C_t, enc_in, seq_len)
        x = x.mean(dim=1)  # 聚合临时通道，回到 (B, enc_in, seq_len)

        x = self.circular_convolution(x, self.w.to(x.device))  # (B, enc_in, seq_len)

        x = self.fc(x)  # (B, enc_in, pred_len)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')  # (B, pred_len, enc_in) restored
        x = z

        return x
