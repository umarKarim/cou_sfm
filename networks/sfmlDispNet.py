from __future__ import absolute_import, division, print_function
import numpy as np
import time 

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_encoder import *


import numpy as np
from collections import OrderedDict

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = []

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs.append(self.alpha * self.sigmoid(self.convs[("dispconv", i)](x)) + self.beta)

        self.outputs = self.outputs[::-1]
        return self.outputs


class DispResNet(nn.Module):

    def __init__(self, num_layers = 18, pretrained = True):
        super(DispResNet, self).__init__()
        self.encoder = ResnetEncoder(num_layers = num_layers, pretrained = pretrained, num_input_images=1)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)

    def init_weights(self):
        pass

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        
        if self.training:
            return outputs[0]
        else:
            return outputs[0]


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = DispResNet().cuda()
    model.train()

    st_time = time.time()
    for i, param in enumerate(model.parameters()):
        curr_param = param.data.reshape(-1).cuda()
        if i == 0:
            param_vec = curr_param
        else:
            param_vec = torch.cat((param_vec, curr_param))
        if curr_param.requires_grad == True:
            print(curr_param.size())
        # print(i, param.size(), param.data.reshape(-1).size())
    print('Parameter vector done')
    print(param_vec.size())
    print('Time taken to populate: {}'.format(time.time() - st_time))
    # param_vec.cuda()

    # update 
    index = 0 
    for i, param in enumerate(model.parameters()):
        curr_param = param.data.reshape(-1)
        # print(curr_param.size())
        param_vec[index: index + curr_param.numel()] += curr_param
        index = curr_param.numel()
    print('time taken to update: {}'.format(time.time() - st_time)) 
    

    '''B = 12

    tgt_img = torch.randn(B, 3, 256, 832).cuda()
    ref_imgs = [torch.randn(B, 3, 256, 832).cuda() for i in range(2)]

    tgt_depth = model(tgt_img)

    print(tgt_depth[0].size())'''


