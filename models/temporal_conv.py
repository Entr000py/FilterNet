import torch
import torch.nn as nn


class DilatedInception(nn.Module):
    """A multi-branch dilated temporal convolution block.

    The implementation is adapted from ``dream_code_v3/3_5_Temporal_conv.py``.
    It keeps the same receptive-field design but adds a safer channel split so
    it can work with arbitrary ``cout``.
    """

    def __init__(self, cin: int, cout: int, dilation_factor: int, seq_len: int):
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        self.seq_len = seq_len
        self.kernel_set = [2, 3, 6, 7]
        self.padding = 0
        self.tconv = nn.ModuleList()

        channel_splits = self._split_channels(cout, len(self.kernel_set))
        active_kernels = []
        for kern, ch in zip(self.kernel_set, channel_splits):
            if ch <= 0:
                continue
            active_kernels.append(kern)
            self.tconv.append(
                nn.Conv2d(cin, ch, (1, kern), dilation=(1, dilation_factor))
            )

        if not self.tconv:
            raise ValueError("cout must be at least 1 to create convolution branches")

        self.kernel_set = active_kernels
        min_kernel = self.kernel_set[-1]
        self.min_time = (
            self.seq_len - dilation_factor * (min_kernel - 1) + self.padding * 2
        )
        if self.min_time <= 0:
            raise ValueError(
                "Invalid configuration: resulting temporal dimension is not positive"
            )

        self.out = nn.Sequential(
            nn.Linear(self.min_time, cin),
            nn.ReLU(),
            nn.Linear(cin, self.seq_len),
        )

    @staticmethod
    def _split_channels(channels: int, num_splits: int):
        base = channels // num_splits
        remainder = channels % num_splits
        splits = []
        for idx in range(num_splits):
            splits.append(base + (1 if idx < remainder else 0))
        return splits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T]
        outputs = [conv(x) for conv in self.tconv]
        min_t = min(branch.size(-1) for branch in outputs)
        outputs = [branch[..., -min_t:] for branch in outputs]
        x = torch.cat(outputs, dim=1)
        x = self.out(x)
        return x


class TemporalConv(nn.Module):
    """Gated temporal convolution block with dilated inception branches."""

    def __init__(self, cin: int, cout: int, dilation_factor: int, seq_len: int):
        super().__init__()
        self.filter_convs = DilatedInception(
            cin=cin, cout=cout, dilation_factor=dilation_factor, seq_len=seq_len
        )
        self.gated_convs = DilatedInception(
            cin=cin, cout=cout, dilation_factor=dilation_factor, seq_len=seq_len
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T]
        filter_out = torch.tanh(self.filter_convs(x))
        gate_out = torch.sigmoid(self.gated_convs(x))
        return filter_out * gate_out
