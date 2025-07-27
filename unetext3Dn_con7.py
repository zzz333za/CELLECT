# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:45:20 2022

@author: zzz333-pc
"""
import torch
import torch.nn as nn
import os
# -----------------------------------------------------------------------------
# Prepare MLP structure for classification
# -----------------------------------------------------------------------------

class EXNet(nn.Module):
    """
    MLP-based model for intra-frame and inter-frame matching.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels per detection.
    n_classes : int
        Number of output classes (e.g. link‐scores or ROI labels).
    base_n_filter : int, optional
        Base filter size for the internal Conv2d layers (default is 8).

    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(EXNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.lrelu = nn.LeakyReLU()
        self.l1 = nn.Linear(self.in_channels * 5, self.in_channels)
        self.l2 = nn.Linear(self.in_channels * 5, self.in_channels)
        self.l3 = nn.Linear(self.in_channels * 5, self.in_channels)
        self.l4 = nn.Linear(3 * 5, self.in_channels)
        self.l5 = nn.Linear(6 * 5, self.in_channels)
        self.l6 = nn.Linear(6 * 5, self.in_channels)
        self.bnx = nn.BatchNorm1d(self.in_channels)
        self.bny = nn.BatchNorm1d(self.in_channels)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.bn2 = nn.BatchNorm1d(self.in_channels)
        self.bn3 = nn.BatchNorm1d(self.in_channels)
        self.bn4 = nn.BatchNorm1d(self.in_channels)
        self.bn5 = nn.BatchNorm1d(self.in_channels)
        self.bn6 = nn.BatchNorm1d(self.in_channels)
        self.bn7 = nn.BatchNorm1d(self.in_channels * 2)
        self.bnc = nn.BatchNorm2d(8)
        self.fc = nn.Linear(self.in_channels * 6, self.in_channels * 2)
        self.fc2 = nn.Linear(self.in_channels * 2, self.in_channels)
        self.out = nn.Linear(self.in_channels, self.n_classes)
        self.c1 = nn.Conv2d(1, 8, kernel_size=(2, 5), stride=1, padding=(0, 2),
                            bias=False)
        self.c2 = nn.Conv2d(1, 8, kernel_size=(2, 3), stride=1, padding=(0, 1),
                            bias=False)
        self.cx = nn.Linear(self.in_channels * 8 * 2, self.in_channels)
        '''self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)'''
        self.outc = nn.Linear(self.in_channels + self.n_classes, 2)
        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.1)
        self.s = nn.Sigmoid()

    def forward(self, x, y, p, s, zs):
        #  Level 1 context pathway
        x01 = torch.cat([x, y[:, 0].unsqueeze(1)], 1)
        x01 = self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x01.unsqueeze(1)))), self.lrelu(
            self.bnc(self.c2(x01.unsqueeze(1))))], 1).reshape(x01.shape[0], -1))
        x02 = torch.cat([x, y[:, 1].unsqueeze(1)], 1)
        x02 = self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x02.unsqueeze(1)))), self.lrelu(
            self.bnc(self.c2(x02.unsqueeze(1))))], 1).reshape(x02.shape[0], -1))
        x03 = torch.cat([x, y[:, 2].unsqueeze(1)], 1)
        x03 = self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x03.unsqueeze(1)))), self.lrelu(
            self.bnc(self.c2(x03.unsqueeze(1))))], 1).reshape(x03.shape[0], -1))
        x04 = torch.cat([x, y[:, 3].unsqueeze(1)], 1)
        x04 = self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x04.unsqueeze(1)))), self.lrelu(
            self.bnc(self.c2(x04.unsqueeze(1))))], 1).reshape(x04.shape[0], -1))
        x05 = torch.cat([x, y[:, 4].unsqueeze(1)], 1)
        x05 = self.cx(torch.cat([self.lrelu(self.bnc(self.c1(x05.unsqueeze(1)))), self.lrelu(
            self.bnc(self.c2(x05.unsqueeze(1))))], 1).reshape(x05.shape[0], -1))
        x1 = torch.cat([x01, x02, x03, x04, x05], 1).reshape(x.shape[0], -1)
        x2 = ((y - x)).reshape(x.shape[0], -1)
        xx = torch.sqrt((x * x).sum(2))
        yy = torch.sqrt((y * y).sum(2))
        xy = x * y
        xy2 = (xx * yy).unsqueeze(-1)
        x3 = xy / xy2
        x3 = x3.reshape(x.shape[0], -1)
        x4 = p.reshape(x.shape[0], -1)
        x5 = s.reshape(x.shape[0], -1)
        x6 = zs.reshape(x.shape[0], -1)
        x1 = self.lrelu(self.bn1(self.l1(x1)))
        x2 = self.lrelu(self.bn2((self.l2(x2))))
        x3 = self.lrelu(self.bn3(self.l3(x3)))
        x4 = self.lrelu(self.bn4((self.l4(x4))))
        x5 = self.lrelu(self.bn5((self.l5(x5))))
        x6 = self.lrelu(self.bn6((self.l6(x6))))
        nx = torch.cat([x1, x2, x3, x4, x5, x6], 1)
        ui = self.bn7(self.lrelu(self.fc(((nx)))))
        nx = self.lrelu(self.fc2(ui))
        f = (self.out(nx))
        return f

class UNet3D(nn.Module):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/pdf/1706.00120.pdf
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=(2,2,1), mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.lnet=LNet3D(6+in_channels,5)
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=(2,2,1), padding=1,
                                   bias=False)
        self.conv3d_c2s1=nn.Conv3d(self.base_n_filter*2, self.base_n_filter * 2, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        self.conv3d_c2s2=nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=1, stride=(2,2,1), padding=0,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=(2,2,1), padding=1,
                                   bias=False)
        self.conv3d_c3s1=nn.Conv3d(self.base_n_filter*4, self.base_n_filter * 4, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        self.conv3d_c3s2=nn.Conv3d(self.base_n_filter*2, self.base_n_filter * 4, kernel_size=1, stride=(2,2,1), padding=0,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=(2,2,1), padding=1,
                                   bias=False)
        self.conv3d_c4s1=nn.Conv3d(self.base_n_filter*8, self.base_n_filter * 8, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        self.conv3d_c4s2=nn.Conv3d(self.base_n_filter*4, self.base_n_filter * 8, kernel_size=1, stride=(2,2,1), padding=0,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=(2,2,1), padding=1,
                                   bias=False)
        self.conv3d_c5s1=nn.Conv3d(self.base_n_filter*16, self.base_n_filter * 16, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        self.conv3d_c5s2=nn.Conv3d(self.base_n_filter*8, self.base_n_filter * 16, kernel_size=1, stride=(2,2,1), padding=0,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.conv_norm_lrelu_f = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 4)
        self.conv3d_f = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter*4 , kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.conv_norm_lrelu_f1 = self.conv_norm_lrelu(self.base_n_filter * 4+ self.n_classes, self.base_n_filter * 4)
        self.conv3d_f1 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter*8 , kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.sigmoid = nn.Sigmoid()

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=(2,2,1), mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        outt = self.conv3d_c2(out)
        outt = self.conv3d_c2s1(outt)
        out=outt+self.conv3d_c2s2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        outt = self.conv3d_c3(out)
        outt = self.conv3d_c3s1(outt)
        out=outt+self.conv3d_c3s2(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        outt = self.conv3d_c4(out)
        outt = self.conv3d_c4s1(outt)
        out=outt+self.conv3d_c4s2(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        outt = self.conv3d_c5(out)
        outt = self.conv3d_c5s1(outt)
        out=outt+self.conv3d_c5s2(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        # print(out.shape)
        # print(context_3.shape)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        f1=self.conv_norm_lrelu_f(out)
        f=self.conv3d_f(f1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out

        
        f=torch.cat([f,ds1_ds2_sum_upscale_ds3_sum_upscale],1)
        f1=self.conv_norm_lrelu_f1(f)
        f1=self.conv3d_f1(f1)
        u=seg_layer[:,:3]
        uu=torch.cat([u,self.softmax(u)],1)
        uu=torch.cat([uu,x],1)
        uo=self.lnet(uu)
     
        ku=self.relu(seg_layer[:,5:])
        return u,uo,f1,seg_layer[:,3:5],ku#,seg_layer[:,4:]
    
    
    
    
    
    
    
class LNet3D(nn.Module):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/pdf/1706.00120.pdf
    """

    def __init__(self, in_channels, n_classes, base_n_filter=2):
        super(LNet3D, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=(2,2,1), mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=(2,2,1), padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=(2,2,1), padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=(2,2,1), padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=(2,2,1), padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)


        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.sigmoid = nn.Sigmoid()

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=(2,2,1), mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        # print(out.shape)
        # print(context_3.shape)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)

        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out

        
        return seg_layer