import torch
import torch.nn as nn
import math


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, causal=True):
        super(Conv, self).__init__()

        self.causal = causal
        if self.causal:
            self.padding = dilation * (kernel_size - 1)
        else:
            self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding is not 0:
            out = out[:, :, :-self.padding]
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv1d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1))

    def forward(self, x):
        out = self.conv(x)
        out = out * torch.exp(self.scale * 3)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, dilation,
                 cin_channels=None, local_conditioning=True, causal=False):
        super(ResBlock, self).__init__()
        self.causal = causal
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.skip = True if skip_channels is not None else False

        self.filter_conv = Conv(in_channels, out_channels, kernel_size, dilation, causal)
        self.gate_conv = Conv(in_channels, out_channels, kernel_size, dilation, causal)
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        if self.skip:
            self.skip_conv = nn.Conv1d(out_channels, skip_channels, kernel_size=1)
            self.skip_conv = nn.utils.weight_norm(self.skip_conv)
            nn.init.kaiming_normal_(self.skip_conv.weight)

        if self.local_conditioning:
            self.filter_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            self.gate_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            self.filter_conv_c = nn.utils.weight_norm(self.filter_conv_c)
            self.gate_conv_c = nn.utils.weight_norm(self.gate_conv_c)
            nn.init.kaiming_normal_(self.filter_conv_c.weight)
            nn.init.kaiming_normal_(self.gate_conv_c.weight)

    def forward(self, tensor, c=None):
        h_filter = self.filter_conv(tensor)
        h_gate = self.gate_conv(tensor)

        if self.local_conditioning:
            h_filter += self.filter_conv_c(c)
            h_gate += self.gate_conv_c(c)

        out = torch.tanh(h_filter) * torch.sigmoid(h_gate)

        res = self.res_conv(out)
        skip = self.skip_conv(out) if self.skip else None
        return (tensor + res) * math.sqrt(0.5), skip


class Wavenet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, num_blocks=1, num_layers=6,
                 residual_channels=256, gate_channels=256, skip_channels=256,
                 kernel_size=3, cin_channels=80, causal=True):
        super(Wavenet, self).__init__()

        self.skip = True if skip_channels is not None else False
        self.front_conv = nn.Sequential(
            Conv(in_channels, residual_channels, 3, causal=causal),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        for b in range(num_blocks):
            for n in range(num_layers):
                self.res_blocks.append(ResBlock(residual_channels, gate_channels, skip_channels,
                                                kernel_size, dilation=2**n,
                                                cin_channels=cin_channels, local_conditioning=True,
                                                causal=causal))

        last_channels = skip_channels if self.skip else residual_channels
        self.final_conv = nn.Sequential(
            nn.ReLU(),
            Conv(last_channels, last_channels, 1, causal=causal),
            nn.ReLU(),
            ZeroConv1d(last_channels, out_channels)
        )

    def forward(self, x, c=None):
        h = self.front_conv(x)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            if self.skip:
                h, s = f(h, c)
                skip += s
            else:
                h, _ = f(h, c)
        if self.skip:
            out = self.final_conv(skip)
        else:
            out = self.final_conv(h)
        return out
