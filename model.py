import torch
from torch import nn
from math import log, pi
from modules import Wavenet
import math

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True, pretrained=False):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))

        self.initialized = pretrained
        self.logdet = logdet

    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        B, _, T = x.size()

        if not self.initialized:
            self.initialize(x)
            self.initialized = True

        log_abs = logabs(self.scale)

        logdet = torch.sum(log_abs) * B * T

        if self.logdet:
            return self.scale * (x + self.loc), logdet

        else:
            return self.scale * (x + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, cin_channel, filter_size=256, num_layer=6, affine=True):
        super().__init__()

        self.affine = affine
        self.net = Wavenet(in_channels=in_channel//2, out_channels=in_channel if self.affine else in_channel//2,
                           num_blocks=1, num_layers=num_layer, residual_channels=filter_size,
                           gate_channels=filter_size, skip_channels=filter_size,
                           kernel_size=3, cin_channels=cin_channel//2, causal=False)

    def forward(self, x, c=None):
        in_a, in_b = x.chunk(2, 1)
        c_a, c_b = c.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a, c_a).chunk(2, 1)

            out_b = (in_b - t) * torch.exp(-log_s)
            logdet = torch.sum(-log_s)
        else:
            net_out = self.net(in_a, c_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output, c=None):
        out_a, out_b = output.chunk(2, 1)
        c_a, c_b = c.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a, c_a).chunk(2, 1)
            in_b = out_b * torch.exp(log_s) + t
        else:
            net_out = self.net(out_a, c_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


def change_order(x, c=None):
    x_a, x_b = x.chunk(2, 1)
    c_a, c_b = c.chunk(2, 1)
    return torch.cat([x_b, x_a], 1), torch.cat([c_b, c_a], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, cin_channel, filter_size, num_layer, affine=True, pretrained=False):
        super().__init__()

        self.actnorm = ActNorm(in_channel, pretrained=pretrained)
        self.coupling = AffineCoupling(in_channel, cin_channel, filter_size=filter_size,
                                       num_layer=num_layer, affine=affine)

    def forward(self, x, c=None):
        out, logdet = self.actnorm(x)
        out, det = self.coupling(out, c)
        out, c = change_order(out, c)

        if det is not None:
            logdet = logdet + det

        return out, c, logdet

    def reverse(self, output, c=None):
        output, c = change_order(output, c)
        x = self.coupling.reverse(output, c)
        x = self.actnorm.reverse(x)
        return x, c


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, cin_channel, n_flow, n_layer, affine=True, pretrained=False, split=False):
        super().__init__()

        self.split = split
        squeeze_dim = in_channel * 2
        squeeze_dim_c = cin_channel * 2

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, squeeze_dim_c, filter_size=256, num_layer=n_layer, affine=affine,
                                   pretrained=pretrained))
        if self.split:
            self.prior = Wavenet(in_channels=squeeze_dim // 2, out_channels=squeeze_dim,
                                 num_blocks=1, num_layers=2, residual_channels=256,
                                 gate_channels=256, skip_channels=256,
                                 kernel_size=3, cin_channels=squeeze_dim_c, causal=False)

    def forward(self, x, c):
        b_size, n_channel, T = x.size()
        squeezed_x = x.view(b_size, n_channel, T // 2, 2).permute(0, 1, 3, 2)
        out = squeezed_x.contiguous().view(b_size, n_channel * 2, T // 2)
        squeezed_c = c.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
        c = squeezed_c.contiguous().view(b_size, -1, T // 2)
        logdet, log_p = 0, 0

        for flow in self.flows:
            out, c, det = flow(out, c)
            logdet = logdet + det
        if self.split:
            out, z = out.chunk(2, 1)
            # WaveNet prior
            mean, log_sd = self.prior(out, c).chunk(2, 1)
            log_p = gaussian_log_p(z, mean, log_sd).sum()
        return out, c, logdet, log_p

    def reverse(self, output, c, eps=None):
        if self.split:
            mean, log_sd = self.prior(output, c).chunk(2, 1)
            z_new = gaussian_sample(eps, mean, log_sd)

            x = torch.cat([output, z_new], 1)
        else:
            x = output

        for flow in self.flows[::-1]:
            x, c = flow.reverse(x, c)

        b_size, n_channel, T = x.size()

        unsqueezed_x = x.view(b_size, n_channel // 2, 2, T).permute(0, 1, 3, 2)
        unsqueezed_x = unsqueezed_x.contiguous().view(b_size, n_channel // 2, T * 2)
        unsqueezed_c = c.view(b_size, -1, 2, T).permute(0, 1, 3, 2)
        unsqueezed_c = unsqueezed_c.contiguous().view(b_size, -1, T * 2)

        return unsqueezed_x, unsqueezed_c


class Flowavenet(nn.Module):
    def __init__(self, in_channel, cin_channel, n_block, n_flow, n_layer, affine=True, pretrained=False,
                 block_per_split=8):
        super().__init__()
        self.block_per_split = block_per_split

        self.blocks = nn.ModuleList()
        self.n_block = n_block
        for i in range(self.n_block):
            split = False if (i + 1) % self.block_per_split or i == self.n_block - 1 else True
            self.blocks.append(Block(in_channel, cin_channel, n_flow, n_layer, affine=affine, 
                                     pretrained=pretrained, split=split))
            cin_channel *= 2
            if not split:
                in_channel *= 2

        self.upsample_conv = nn.ModuleList()
        for s in [16, 16]:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

    def forward(self, x, c):
        B, _, T = x.size()
        logdet, log_p_sum = 0, 0
        out = x
        c = self.upsample(c)
        for block in self.blocks:
            out, c, logdet_new, logp_new = block(out, c)
            logdet = logdet + logdet_new
            log_p_sum = log_p_sum + logp_new
        log_p_sum += 0.5 * (- log(2.0 * pi) - out.pow(2)).sum()
        logdet = logdet / (B * T)
        log_p = log_p_sum / (B * T)
        return log_p, logdet

    def reverse(self, z, c):
        _, _, T = z.size()
        _, _, t_c = c.size()
        if T != t_c:
            c = self.upsample(c)
        z_list = []
        x = z
        for i in range(self.n_block):
            b_size, _, T = x.size()
            squeezed_x = x.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
            x = squeezed_x.contiguous().view(b_size, -1, T // 2)
            squeezed_c = c.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
            c = squeezed_c.contiguous().view(b_size, -1, T // 2)
            if not ((i + 1) % self.block_per_split or i == self.n_block - 1):
                x, z = x.chunk(2, 1)
                z_list.append(z)

        for i, block in enumerate(self.blocks[::-1]):
            index = self.n_block - i
            if not (index % self.block_per_split or index == self.n_block):
                x, c = block.reverse(x, c, z_list[index // self.block_per_split - 1])
            else:
                x, c = block.reverse(x, c)
        return x

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c
