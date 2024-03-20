from functools import partial

import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from torchvision import models

torch.backends.cudnn.enabled = True

nonlinearity = partial(F.relu, inplace=True)



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class E_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=5, act_layer=nn.ReLU6, drop=0.):
        super(E_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.conv1 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=ksize,
                                groups=hidden_features)
        self.conv2 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3,
                                groups=hidden_features)
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.fc2(x1 + x2)
        x = self.act(x)

        return x


def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)


class PAM_Module(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.softplus_feature = softplus_feature_map
        self.eps = eps

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
        self.channel = E_FFN(in_features=in_places)
        self.channel_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        # x_channel:torch.Size([2, 2048, 16, 16])
        x_channel = self.channel(x)
        # x_channel = self.channel_conv(x_channel)
        # print('x_channel:{}'.format(x_channel.shape))
        # print('Q_test:{}'.format(Q_test.shape))x_sum = Q_test + x_channel
        # print('x_sum:{}'.format(x_sum.shape))
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        # print('Q的形状：{}'.format(Q.shape))
        K = self.key_conv(x).view(batch_size, -1, width * height)
        # print('K的形状：{}'.format(K.shape))
        v_test = self.value_conv(x)
        v_sum = v_test + x_channel
        # print('v_test:{}'.format(v_test.shape))
        # V = self.value_conv(x).view(batch_size, -1, width * height)
        V = v_sum.view(batch_size, -1, width * height)
        # print('V的形状：{}'.format(V.shape))
        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        # print('Q的形状：{}'.format(Q.shape))
        K = self.softplus_feature(K)
        # print('K的形状：{}'.format(K.shape))
        KV = torch.einsum("bmn, bcn->bmc", K, V)
        # print('KV的形状：{}'.format(KV.shape))
        # z = torch.sum(K,dim=-1)+self.eps
        # print('z的 shape:{}'.format(z.shape))
        # torch.sum(K, dim=-1):[1,8]
        # torch.sum(K, dim=-1) + self.eps:[1,8]
        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)
        # weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        return (x + self.gamma * weight_value).contiguous()


class CAM_Module(nn.Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out


# class PAM_CAM_Layer(nn.Module):
#     def __init__(self, in_ch):
#         super(PAM_CAM_Layer, self).__init__()
#         self.PAM = PAM_Module(in_ch)
#         self.CAM = CAM_Module()
#
#     def forward(self, x):
#         return self.PAM(x) + self.CAM(x)

class PAM_CAM(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_CAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.softplus_feature = softplus_feature_map
        self.eps = eps

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
        self.channel = E_FFN(in_features=in_places)
        self.channel_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.conv_2 = nn.Conv2d(2 * in_places, in_places, 1, 1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        # PAM
        batch_size, chnnels, width, height = x.shape
        # x_channel:torch.Size([2, 2048, 16, 16])
        x_channel = self.channel(x)
        # x_channel = self.channel_conv(x_channel)
        # print('x_channel:{}'.format(x_channel.shape))
        # print('Q_test:{}'.format(Q_test.shape))x_sum = Q_test + x_channel
        # print('x_sum:{}'.format(x_sum.shape))
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        # print('Q的形状：{}'.format(Q.shape))
        K = self.key_conv(x).view(batch_size, -1, width * height)
        # print('K的形状：{}'.format(K.shape))
        v_test = self.value_conv(x)
        v_sum = v_test + x_channel
        # print('v_test:{}'.format(v_test.shape))
        # V = self.value_conv(x).view(batch_size, -1, width * height)
        V = v_sum.view(batch_size, -1, width * height)
        # print('V的形状：{}'.format(V.shape))
        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        # print('Q的形状：{}'.format(Q.shape))
        K = self.softplus_feature(K)
        # print('K的形状：{}'.format(K.shape))
        KV = torch.einsum("bmn, bcn->bmc", K, V)
        # print('KV的形状：{}'.format(KV.shape))
        # z = torch.sum(K,dim=-1)+self.eps
        # print('z的 shape:{}'.format(z.shape))
        # torch.sum(K, dim=-1):[1,8]
        # torch.sum(K, dim=-1) + self.eps:[1,8]
        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)
        # weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        # out_pam = (x + self.gamma * weight_value).contiguous()

        # CAM
        proj_query = x.view(batch_size, chnnels, -1)
        # print('proj_query:{}'.format(proj_query.shape))
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        # print('proj_key:{}'.format(proj_key.shape))
        energy = torch.bmm(proj_query, proj_key)
        # print('energy:{}'.format(energy.shape))
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # print('energy_new:{}'.format(energy_new.shape))
        attention = self.softmax(energy_new)
        # print('atention:{}'.format(attention.shape))
        # proj_value = x.view(batch_size, chnnels, -1)
        # print('proj_value:{}'.format(proj_value.shape))
        out = torch.bmm(attention, V)
        out_cam = out.view(batch_size, chnnels, height, width)
        out_cat = torch.cat([weight_value, out_cam], dim=1)
        out_cat = self.conv_2(out_cat)
        out = self.gamma * out_cat + x

        return out


class DWconv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1, kernel_size=3):
        super(DWconv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


# class FreBlock(nn.Module):
#     def __init__(self):
#         super(FreBlock, self).__init__()
#
#     def forward(self, x):
#         x = x + 1e-8
#         mag = torch.abs(x)
#         pha = torch.angle(x)
#
#         return mag, pha
#
#
# class Fre_Spa(nn.Module):
#     def __init__(self, n_feats):
#         super().__init__()
#         self.n_feats = n_feats
#         self.Conv1 = nn.Sequential(
#             nn.Conv2d(n_feats, 2 * n_feats, 7, 1, 3),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0))
#
#         self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
#         self.FF = FreBlock()
#         self.conv_original = nn.Conv2d(2 * n_feats, n_feats, kernel_size=1)
#         # e1:torch.Size([2, 256, 128, 128])
#         # e2:torch.Size([2, 512, 64, 64])
#         # e3:torch.Size([2, 1024, 32, 32])
#         # e4:torch.Size([2, 2048, 16, 16])
#         self.dw1 = DWconv(in_ch=n_feats, out_ch=n_feats, padding=8, dilation=8, kernel_size=3)
#         self.dw2 = DWconv(in_ch=n_feats, out_ch=n_feats, padding=9, dilation=3, kernel_size=7)
#         self.pam_change = PAM_Module(in_places=n_feats)
#         self.conv = nn.Conv2d(2 * n_feats, n_feats, 1, 1)
#
#     def forward(self, x, y):
#         b, c, H, W = x.shape
#
#         a = 0.5
#         mix = x + y
#
#         mix_mag, mix_pha = self.FF(mix)
#         mix_pha = self.Conv1(mix_pha)
#         mix_mag = self.pam_change(mix_mag)
#
#         mix_pha1 = self.dw1(mix_pha)
#         mix_pha2 = self.dw2(mix_pha)
#         mix_pha = self.conv(torch.concat([mix_pha1, mix_pha2], dim=1))
#         mix_pha = self.dw1(mix_pha)
#         mix_pha = self.dw2(mix_pha)
#
#         real_main = mix_mag * torch.cos(mix_pha)
#         imag_main = mix_mag * torch.sin(mix_pha)
#         x_out_main = torch.complex(real_main, imag_main)
#         x_out_main = torch.abs(torch.fft.irfft2(x_out_main, s=(H, W), norm='backward')) + 1e-8
#
#         return self.Conv2(a * x_out_main + (1 - a) * mix)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class BE_Model(nn.Module):
    def __init__(self, in_ch_high, in_ch_low, out_ch, out_HW):
        """
           Args:
               in_ch_high : Number of high feature map input channels
               in_ch_low : Number of low feature  map input channels
               out_ch : Number of output channels
               out_HW : The height and width of the output
        """
        super(BE_Model, self).__init__()
        kernel_size, stride, padding = 3, 1, 1
        self.out_HW = out_HW
        self.DFconv1 = nn.Sequential(
            DeformConv2d(in_ch_high, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.DFconv2 = nn.Sequential(
            DeformConv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.decoderB = DecoderBlock(in_ch_low, out_ch)

        # x :[B,2048,16,16]             y :[B,64,128,128]
        # x: high features  y : low features

    def forward(self, x, y):
        # [2,256,16,16]
        e4_256_16_16 = self.DFconv1(x)
        # [2,256,32,32]
        e4_256_32_32 = F.interpolate(e4_256_16_16, (32, 32))
        e4_256_32_32 = self.DFconv2(e4_256_32_32)
        # [2,256,256,256]
        e4_256_256_256 = F.interpolate(e4_256_32_32, (self.out_HW, self.out_HW))
        # [2,256,512,512]
        y_256_256_256 = self.decoderB(y)
        y_256_256_256 = F.interpolate(y_256_256_256, (256, 256), mode='bilinear', align_corners=True)
        if self.out_HW != 256:
            y_256_256_256 = F.interpolate(e4_256_256_256, (self.out_HW, self.out_HW))

        boundary = y_256_256_256 - e4_256_256_256

        return boundary


class MANet(nn.Module):
    def __init__(self, num_classes=7):
        super(MANet, self).__init__()
        self.name = 'MANet'

        filters = [256, 512, 1024, 2048]

        resnet = models.resnet50(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)


        # ResNet的 Layer1 的结果经过三个空洞卷积
        self.convl1_1 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, padding=6, dilation=6, bias=True),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.convl1_2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 4, padding=12, dilation=12, bias=True),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.convl1_3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 8, padding=18, dilation=18, bias=True),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        # ResNet的 Layer2 的结果经过2个空洞卷积:dilation = 6 , dilation = 12
        self.convl2_1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 2, padding=6, dilation=6, bias=True),
            nn.BatchNorm2d(1024, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.convl2_2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 4, padding=12, dilation=12, bias=True),
            nn.BatchNorm2d(1024, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.convl3 = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, 2, padding=6, dilation=6, bias=True),
            nn.BatchNorm2d(2048, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.convl4 = nn.Conv2d(2048, 2048, 1)

        self.conv_c1_2 = nn.Conv2d(512, 1024, kernel_size=1)
        self.conv_c1_3 = nn.Conv2d(512, 2048, kernel_size=1)
        self.conv_c2_2 = nn.Conv2d(1024, 2048, kernel_size=1)

        self.pcam1 = PAM_CAM(in_places=256)
        self.pcam2 = PAM_CAM(in_places=512)
        self.pcam3 = PAM_CAM(in_places=1024)
        self.pcam4 = PAM_CAM(in_places=2048)

        self.be_main = BE_Model(in_ch_high=2048, in_ch_low=256, out_ch=256, out_HW=256)
        self.be_aux = BE_Model(in_ch_high=1024, in_ch_low=256, out_ch=256, out_HW=256)
        self.be_aux2 = BE_Model(in_ch_high=512, in_ch_low=256, out_ch=256, out_HW=256)

    def forward(self, x):
        # Encoder
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)



        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        boundary_be_main = self.be_main(e4, e1)
        boundary_be_aux = self.be_aux(e3, e1)
        boundary_be_aux2 = self.be_aux2(e2,e1)


        # c1_1,dilation=6:torch.Size([2, 512, 64, 64])
        c1_1 = self.convl1_1(e1)
        c1_2 = self.convl1_2(e1)
        # c1_2:torch.Size([2, 1024, 32, 32])
        c1_2 = self.conv_c1_2(c1_2)
        c1_3 = self.convl1_3(e1)
        # c1_3:torch.Size([2, 2048, 16, 16])
        c1_3 = self.conv_c1_3(c1_3)
        # c2_1:torch.Size([2, 1024, 32, 32])
        c2_1 = self.convl2_1(e2)
        # c2_2:torch.Size([2, 2048, 16, 16])
        c2_2 = self.convl2_2(e2)
        c2_2 = self.conv_c2_2(c2_2)
        # c3:torch.Size([2, 2048, 16, 16])
        c3 = self.convl3(e3)
        # c4:torch.Size([2, 2048, 16, 16])
        c4 = self.convl4(e4)
        # torch.Size([2, 512, 64, 64])
        e2 = e2 + c1_1
        # torch.Size([2, 1024, 32, 32])
        e3 = e3 + c2_1 + c1_2
        # torch.Size([2, 2048, 16, 16])
        e4 = c4 + c1_3 + c2_2 + c3

        pcam_e4 = self.pcam4(e4)
        pcam_e3 = self.pcam3(e3)
        pcam_e2 = self.pcam2(e2)
        pcam_e1 = self.pcam1(e1)
        # Decoder
        # d4的shapetorch.Size([2, 1024, 32, 32])
        d4 = self.decoder4(pcam_e4) + pcam_e3  # +self.sm_e3(e3)
        # d3的shapetorch.Size([2, 512, 64, 64])
        d3 = self.decoder3(d4) + pcam_e2  # +self.sm_e2(e2)
        # d2的shapetorch.Size([2, 256, 128, 128])
        d2 = self.decoder2(d3) + pcam_e1  # +self.sm_e1(e1)
        # d1的torch.Size([2, 256, 256, 256])
        d1 = self.decoder1(d2) + boundary_be_main + boundary_be_aux + boundary_be_aux2
        # print('d1的{}'.format(d1.shape))

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out
from torchstat import stat

if __name__ == '__main__':
    num_classes = 7
    in_batch, inchannel, in_h, in_w = 2, 3, 512, 512
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = MANet(num_classes)
    out = net(x)
    # stat(net, (3, 512, 512))
    print(out.shape)
# from torchstat import stat
# model = MANet(num_classes=7)
# stat(model, input_size=(3, 512, 512))
# from torchsummary import summary
#
# model = MANet(num_classes=7)
# summary(model, input_size=(3, 512, 512))
