import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    pad = int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)
    return pad

def pad_to(x, stride):
    h, w, d = x.shape[-3:]
    # print(h,w,d)

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    if d % stride > 0:
        new_d = d + stride - d % stride
    else:
        new_d = d
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    ld, ud = int((new_d - d) / 2), int(new_d - d) - int((new_d - d) / 2)
    pads = (ld, ud, lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)
    return out, pads

def unpad(x, pad):
    if pad[4]+pad[5] > 0:
        x = x[:,:,pad[4]:-pad[5],:,:]
    if pad[2]+pad[3] > 0:
        x = x[:,:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,:,pad[0]:-pad[1]]
    return x

def max_pooling_3d(ks=(2, 2, 2), stride=(2, 2, 2)):
    return nn.MaxPool3d(kernel_size=ks, stride=stride, padding=0)


class BAGConv(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), use_gate=True):
        super(BAGConv, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm3d = torch.nn.BatchNorm3d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.use_gate = use_gate

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv3d(input)
        mask = self.mask_conv3d(input)

        # gated features
        if self.batch_norm:
            x = self.batch_norm3d(x)

        if self.activation is not None:
            if self.use_gate:
                x = self.activation(x) * self.gated(mask)
            else:
                x = self.activation(x)
        else:
            if self.use_gate:
                x = x * self.gated(mask)
        return x


class BAGDeConv(torch.nn.Module):

    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(BAGDeConv, self).__init__()
        self.conv3d = BAGConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation, use_gate=True)
        self.scale_factor = scale_factor

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        x = self.conv3d(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, in_dim=1, out_dim=2, num_filters=32, batch_norm=True, cnum=32, use_gate=True):
        super(Decoder, self).__init__()
        div = 4
        activation = nn.LeakyReLU(0.2, inplace=True)
        self.dec1_1 = BAGDeConv(2, 16 * cnum, 8 * cnum, 3, 1, padding=get_pad(32 // div, 3, 1), batch_norm=batch_norm,
                                       activation=activation)
        self.dec1_2 = BAGConv(2*8 * cnum, 8 * cnum, 3, 1, padding=get_pad(32 // div, 3, 1), batch_norm=batch_norm,
                                   activation=activation, use_gate=True)

        self.dec2_1 = BAGDeConv(2, 8 * cnum, 4 * cnum, 3, 1, padding=get_pad(64 // div, 3, 1), batch_norm=batch_norm,
                                       activation=activation)
        self.dec2_2 = BAGConv(2*4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64 // div, 3, 1), batch_norm=batch_norm,
                                   activation=activation, use_gate=True)

        self.dec3_1 = BAGDeConv(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128 // div, 3, 1), batch_norm=batch_norm,
                                       activation=activation)
        self.dec3_2 = BAGConv(2*2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128 // div, 3, 1), batch_norm=batch_norm,
                                   activation=activation, use_gate=True)

        self.dec4_1 = BAGDeConv(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm,
                                       activation=activation)
        self.dec4_2 = BAGConv(2 * cnum, cnum, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm,
                                   activation=activation, use_gate=True)

        self.out = BAGConv(cnum, out_dim, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm,
                                  activation=None, use_gate=True)

    def forward(self, input, skip):
        # Up sampling
        down_4 = skip[3]
        down_3 = skip[2]
        down_2 = skip[1]
        down_1 = skip[0]

        trans_1 = self.dec1_1(input)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.dec1_2(concat_1)

        trans_2 = self.dec2_1(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.dec2_2(concat_2)

        trans_3 = self.dec3_1(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.dec3_2(concat_3)

        trans_4 = self.dec4_1(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.dec4_2(concat_4)

        x = self.out(up_4)
        return x


class TWBAG(nn.Module):
    def __init__(self, in_dim=1, out_dim=2, batch_norm=True, cnum=32):
        super(TWBAG, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # activation = nn.ReLU(inplace=True)
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        div = 4
        self.enc1_1 = BAGConv(in_dim, cnum, 3, 1, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm,
                                     activation=activation, use_gate=True)
        self.enc1_2 = BAGConv(cnum, cnum, 3, 2, padding=get_pad(256 // div, 3, 1), batch_norm=batch_norm,
                                     activation=activation, use_gate=True)
        # downsample to 128
        self.enc2_1 = BAGConv(cnum, 2 * cnum, 3, 1, padding=get_pad(128 // div, 3, 1), batch_norm=batch_norm,
                                     activation=activation, use_gate=True)
        self.enc2_2 = BAGConv(2 * cnum, 2 * cnum, 3, 2, padding=get_pad(128 // div, 3, 1), batch_norm=batch_norm,
                                     activation=activation, use_gate=True)
        # downsample to 64
        self.enc3_1 = BAGConv(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(64 // div, 3, 1), batch_norm=batch_norm,
                                     activation=activation, use_gate=True)
        self.enc3_2 = BAGConv(4 * cnum, 4 * cnum, 3, 2, padding=get_pad(64 // div, 3, 1), batch_norm=batch_norm,
                                     activation=activation, use_gate=True)
        # downsample to 32
        self.enc4_1 = BAGConv(4 * cnum, 8 * cnum, 3, 1, padding=get_pad(32 // div, 3, 1), batch_norm=batch_norm,
                                     activation=activation, use_gate=True)
        self.enc4_2 = BAGConv(8 * cnum, 8 * cnum, 3, 2, padding=get_pad(32 // div, 3, 1), batch_norm=batch_norm,
                                     activation=activation, use_gate=True)

        # Bridge
        self.bridge = BAGConv(8 * cnum, 16 * cnum, 3, 1, padding=get_pad(16 // div, 3, 1), batch_norm=batch_norm,
                                     activation=activation, use_gate=True)

        # tensor-wise decoding
        self.dec1 = Decoder(16 * cnum, 1)
        self.dec2 = Decoder(16 * cnum, 1)
        self.dec3 = Decoder(16 * cnum, 1)
        self.dec4 = Decoder(16 * cnum, 1)
        self.dec5 = Decoder(16 * cnum, 1)
        self.dec6 = Decoder(16 * cnum, 1)


    def forward(self, x, save_feat=False):
        skip = []

        x, pads = pad_to(x, 16)  # Padded data, feed this to your network
        # Down sampling
        down_1 = self.enc1_1(x)
        pool_1 = self.enc1_2(down_1)
        skip.append(down_1)

        down_2 = self.enc2_1(pool_1)
        pool_2 = self.enc2_2(down_2)
        x = pool_2
        skip.append(down_2)

        down_3 = self.enc3_1(pool_2)
        pool_3 = self.enc3_2(down_3)
        x = pool_3
        skip.append(down_3)

        down_4 = self.enc4_1(pool_3)
        pool_4 = self.enc4_2(down_4)
        x = pool_4
        skip.append(down_4)

        # Bridge
        bridge = self.bridge(pool_4)

        # decoding
        up1 = self.dec1(bridge, skip)
        up2 = self.dec2(bridge, skip)
        up3 = self.dec3(bridge, skip)
        up4 = self.dec4(bridge, skip)
        up5 = self.dec5(bridge, skip)
        up6 = self.dec6(bridge, skip)

        out = torch.cat([up1, up2, up3, up4, up5, up6], dim=1)
        out = unpad(out, pads)
        return out