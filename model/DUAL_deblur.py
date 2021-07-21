import torch
import torch.nn as nn
import model.block as B
from model.arch_util import ResidualBlockNoBN
from model.common import MeanShift

def make_model(opt):
    model = RFDN()
    return model


class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


class PredeblurModule(nn.Module):
    """Pre-dublur module.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in

        self.conv_first = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        if self.hr_in:
            # downsample x4 by stride conv
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        # generate feature pyramid
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList(
            [ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat_l1 = self.lrelu(self.conv_first(x))
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))

        # generate feature pyramid
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))

        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))

        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1



class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)
        self.encode = PredeblurModule(3,64,hr_in=True)
        self.fea_conv = B.conv_layer(64, nf, kernel_size=3)

        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        self.final_lym = B.conv_layer(nf,3,kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0
        self.lam = LAM_Module(nf)
        self.scam = CSAM_Module(nf)
        self.last = B.conv_layer(nf*2, nf, kernel_size=3)
        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

    def forward(self, input):
        input =self.sub_mean(input)
        input = self.encode(input)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        b1s = out_B1.unsqueeze(1)
        #print(b1s.size())
        b2s = out_B2.unsqueeze(1)
        b3s = out_B3.unsqueeze(1)
        b4s = out_B4.unsqueeze(1)
        out_a = torch.cat([b1s,b2s,b3s,b4s],1)
        out_b = out_B4
        res_a = self.lam(out_a)
        res_a = self.c(res_a)
        #print(res_a.size())
        res_b = self.scam(out_b)
        #print(res_b.size())
        #print(res_a.size())
        out = torch.cat([res_a,res_b],1)
        out = self.last(out)
        out += out_fea


        #out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        #out_lr = self.LR_conv(out_B) + out_fea
        #output =self.final_lym(out_lr)
        output = self.upsampler(out)
        output = self.add_mean(output)
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx



#
# class DUAL_deblur(nn.Module):
#     def __init__(self, opt, conv=common.default_conv):
#         super(DUAL_deblur, self).__init__()
#         self.opt = opt
#         self.scale = [2,4]
#         self.phase = 2
#         n_blocks = opt.n_blocks
#         n_feats = opt.n_feats
#         kernel_size = 3
#
#         act = nn.ReLU(True)
#
#         self.upsample = nn.Upsample(scale_factor=max(self.scale),
#                                     mode='bicubic', align_corners=False)
#
#         rgb_mean = (0.4488, 0.4371, 0.4040)
#         rgb_std = (1.0, 1.0, 1.0)
#         self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)
#
#         self.head = conv(opt.n_colors, n_feats, kernel_size)
#
#         self.down = [
#             common.DownBlock(opt, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
#             ) for p in range(self.phase)
#         ]
#
#         self.down = nn.ModuleList(self.down)
#
#         up_body_blocks = [[
#             common.RCAB(
#                 conv, n_feats * pow(2, p), kernel_size, act=act
#             ) for _ in range(n_blocks)
#         ] for p in range(self.phase, 1, -1)
#         ]
#
#         up_body_blocks.insert(0, [
#             common.RCAB(
#                 conv, n_feats * pow(2, self.phase), kernel_size, act=act
#             ) for _ in range(n_blocks)
#         ])
#
#         # The fisrt upsample block
#         up = [[
#             common.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
#             conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
#         ]]
#
#         # The rest upsample blocks
#         for p in range(self.phase - 1, 0, -1):
#             up.append([
#                 common.Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False),
#                 conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
#             ])
#
#         self.up_blocks = nn.ModuleList()
#         for idx in range(self.phase):
#             self.up_blocks.append(
#                 nn.Sequential(*up_body_blocks[idx], *up[idx])
#             )
#
#         # tail conv that output sr imgs
#         tail = [conv(n_feats * pow(2, self.phase), opt.n_colors, kernel_size)]
#         for p in range(self.phase, 0, -1):
#             tail.append(
#                 conv(n_feats * pow(2, p), opt.n_colors, kernel_size)
#             )
#         self.tail = nn.ModuleList(tail)
#
#         self.add_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)
#
#     def forward(self, x):
#         # upsample x to target sr size
#         # x = self.upsample(x)
#         x = self.sub_mean(x)
#
#         x = self.head(x)
#
#         # down phases,
#         copies = []
#         for idx in range(self.phase):
#             copies.append(x)
#             x = self.down[idx](x)
#         # up phases
#         # sr = self.tail[0](x)
#         # sr = self.add_mean(sr)
#         # results = [sr]
#         for idx in range(self.phase):
#             # upsample to SR features
#             x = self.up_blocks[idx](x)
#             # concat down features and upsample features
#             x = torch.cat((x, copies[self.phase - idx - 1]), 1)
#             # output sr imgs
#             sr = self.tail[idx + 1](x)
#
#             sr = self.add_mean(sr)
#             # print(sr)
#
#             # results.append(sr)
#
#         return sr
