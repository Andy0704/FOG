import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


def receptive_field(kernel_size, dilations, n_conv_per_block=2):
    return 1 + n_conv_per_block * sum((kernel_size - 1) * d for d in dilations)

class Chomp1d(nn.Module):
    """專門用來切除右側 Padding，確保時間上的因果關係"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一組卷積
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(dropout) # 加入 Dropout1d 

        # 第二組卷積
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout1d(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        
        # 殘差連接：如果通道數不同，需用 1x1 卷積調整維度
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) # 核心：殘差相加
    
class BasicTCN(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, num_channels=[32, 64, 128], kernel_size=5,
                 dropout=0.2, dilations=None, window_size=64):
        super(BasicTCN, self).__init__()
        num_levels = len(num_channels)
        if dilations is None:
            dilations = [2 ** i for i in range(num_levels)]

        rf = receptive_field(kernel_size, dilations, 2)
        if rf > window_size:
            raise ValueError(f"RF={rf} > window_size={window_size} "
                              f"(K={kernel_size}, dilations={dilations}); skip this combo.")
        print(f"[RF] {rf} (<= {window_size})")

        layers = []
        for i in range(num_levels):
            dilation_size = dilations[i]
            in_channels = n_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], n_classes)

    def forward(self, x):
        x = self.network(x)
        x = x[:, :, -1] # 提取最後一個時間步的特徵
        return self.fc(x)