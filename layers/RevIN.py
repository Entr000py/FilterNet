import torch
import torch.nn as nn

class RevIN(nn.Module):
    # RevIN 在时间维上隔离统计量，提供前向/后向规范化
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: 通道数
        :param eps: 为数值稳定额外加入的小常数
        :param affine: True 时会学习仿射变换参数
        """
        super(RevIN, self).__init__()
        self.num_features = num_features  # 通道数，后续需要对每个通道独立归一化
        self.eps = eps  # 防止除零的小数
        self.affine = affine  # 是否学习位移与缩放
        self.subtract_last = subtract_last  # 是否用最后一条时间步代替均值
        self._stats_ready = False  # 标记当前是否已有可用统计量
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)  # 先计算均值与标准差供标准化使用
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)  # 反向映射回原始分布
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # 初始化每个通道的仿射变换参数
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))    # 全1向量
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))     # 全0向量

    def _get_affine_weight(self):
        # 将过小的仿射权重轻微偏移，保证正/反向使用同一数值且可逆
        weight = self.affine_weight
        offset = torch.where(weight >= 0, torch.full_like(weight, self.eps), -torch.full_like(weight, self.eps))
        safe = torch.where(torch.abs(weight) > self.eps, weight, weight + offset)
        return safe

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))  # 除 batch 和 feature 外的维度
        if self.subtract_last:
            self.last = x[:, -1:, :].detach()  # 用最后一帧作为参考值（与输入图断开）
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()  # 记录均值
        var = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()  # 标准差
        self._stats_ready = True

    def _normalize(self, x):
        if not self._stats_ready:
            raise RuntimeError('RevIN: 还没有可用的统计量，请先运行 mode="norm"')
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            weight = self._get_affine_weight()
            x = x * weight
            x = x + self.affine_bias
        # 以上操作每个通道独立执行，保持形状一致
        return x

    def _denormalize(self, x):
        if not self._stats_ready:
            raise RuntimeError('RevIN: 还没有可用的统计量，请先运行 mode="norm"')
        if self.affine:
            weight = self._get_affine_weight()
            x = x - self.affine_bias
            x = x / weight
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        # 还原每个通道的原始分布
        return x
