# from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
import random

random.seed(1)

def normalize(x, a, b, c, d):
    '''
    x from (a, b) to (c, d)
    '''
    return (float(x) - a) * (float(d) - c) / (float(b) - a) + float(c)

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    """
    train_step1: self.step_flag=1, param_list needs value, param_channels require
        no grad while UNet requires grad
    test_step1: self.step_flag=2, param_list needs value
    train_step2: self.step_flag=3, param_list is none, param_channels require grad
        while UNet requires no grad
    test_step2: self.step_flag=4, param_list is a trained pickle
    """
    
    def __init__(self, in_ch=3, out_ch=3, step_flag=1, img_size=256):
        super(U_Net, self).__init__()
        
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.img_size = img_size
        
        # the added channels
        param_num = 5
        
        self.step_flag = step_flag

        self.Avgpool  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = conv_block(in_ch + param_num, filters[0])
        self.Conv1_2 = conv_block(filters[0], filters[0])
        self.Conv2_1 = conv_block(filters[0], filters[1])
        self.Conv2_2 = conv_block(filters[1] + param_num, filters[1])
        self.Conv3_1 = conv_block(filters[1], filters[2])
        self.Conv3_2 = conv_block(filters[2] + param_num, filters[2])
        self.Conv4_1 = conv_block(filters[2], filters[3])
        self.Conv4_2 = conv_block(filters[3] + param_num, filters[3])
        self.Conv5_1 = conv_block(filters[3], filters[4])
        self.Conv5_2 = conv_block(filters[4] + param_num, filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5_1 = conv_block(filters[4], filters[3])
        self.Up_conv5_2 = conv_block(filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4_1 = conv_block(filters[3], filters[2])
        self.Up_conv4_2 = conv_block(filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3_1 = conv_block(filters[2], filters[1])
        self.Up_conv3_2 = conv_block(filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2_1 = conv_block(filters[1], filters[0])
        self.Up_conv2_2 = conv_block(filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
    
        if self.step_flag == 1:
            for p in self.parameters():
                p.requires_grad = True
                    
        elif self.step_flag > 1:
            for p in self.parameters():
                p.requires_grad = False

        # cff, n1, cspace, wtransform, neighborhood

        self.param_layer = torch.cat([
            torch.rand([1]).expand(1, 1, self.img_size, self.img_size),
            torch.tensor(normalize(torch.randint(1, 3, [1]) * 4., 4, 8, 0, 1)).expand(1, 1, self.img_size, self.img_size),
            torch.tensor(normalize(torch.randint(0, 2, [1]), 0, 1, 0, 1)).expand(1, 1, self.img_size, self.img_size),
            torch.tensor(normalize(torch.randint(0, 2, [1]), 0, 1, 0, 1)).expand(1, 1, self.img_size, self.img_size),
            torch.tensor(normalize(torch.randint(4, 16, [1]), 4, 15, 0, 1)).expand(1, 1, self.img_size, self.img_size)
        ], dim=1).cuda()
        # print(self.param_layer)

        # self.param_layer = torch.cat([
        #     torch.ones(1, 1, self.img_size, self.img_size).float() / 2.,
        #     torch.ones(1, 1, self.img_size, self.img_size).float() / 2.,
        #     torch.ones(1, 1, self.img_size, self.img_size).float() / 2.,
        #     torch.ones(1, 1, self.img_size, self.img_size).float() / 2.,
        #     torch.ones(1, 1, self.img_size, self.img_size).float() / 2.
        # ], dim=1).cuda()

        # self.param_layer = torch.cat([
        #     (torch.rand(1, 1, self.img_size, self.img_size).float()*14 + 1 - 0.) * (1. - 0.) / (15. - 0.) + 0.,
        #     (torch.randint(1, 3, (1, 1, self.img_size, self.img_size)).float()*4 - 4.) * (1. -0.) / (8. - 4.) + 0.,
        #     (torch.randint(0, 2, (1, 1, self.img_size, self.img_size)).float() - 0.) * (1. - 0.) / (1. - 0.) + 0.,
        #     (torch.randint(0, 2, (1, 1, self.img_size, self.img_size)).float() - 0.) * (1. - 0.) / (1. - 0.) + 0.,
        #     (torch.randint(4, 13, (1, 1, self.img_size, self.img_size)).float() - 4.) * (1. - 0.) / (12. - 4.) + 0.
        # ], dim=1).cuda()

        
        if self.step_flag == 3:
            self.param_layer.requires_grad = True
        elif self.step_flag <= 2 or step_flag > 3:
            self.param_layer.requires_grad = False
    
    def return_param_layer(self):
        return self.param_layer

    def load_param_layer(self, value):
        self.param_layer = value
    
    def return_param_value(self):
        res = []
        for idx in range(self.param_layer.shape[1]):
            res.append(self.param_layer[0, idx, :, :].cpu().detach().numpy().mean())
        return np.array(res)
    
    def update_param(self):
        self.param_layer.requires_grad=False
        self.param_layer = torch.clamp(self.param_layer.clone(), 0, 1)
        self.param_layer.requires_grad=True

    def forward(self, x, param_list=None):
        
        if self.step_flag <= 2:
            param_list = param_list.view(param_list.size(0), param_list.size(1), 1, 1)
            param_list = param_list.repeat(1, 1, x.size(2), x.size(3))
            param_layer = param_list
            # print(param_layer)
        
        elif self.step_flag > 2:
            pass

        if self.step_flag <= 2:
            e1 = self.Conv1_1(torch.cat([x, param_layer], dim=1))
        elif self.step_flag > 2:
            e1 = self.Conv1_1(torch.cat([x, self.param_layer.repeat(x.size(0), 1, 1, 1)], dim=1))
        e1 = self.Conv1_2(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2_1(e2)
        if self.step_flag <= 2:
            e2 = self.Conv2_2(torch.cat([e2, self.Avgpool(param_layer)], dim=1))
        elif self.step_flag > 2:
            e2 = self.Conv2_2(torch.cat([e2, self.Avgpool(
                self.param_layer.repeat(x.size(0), 1, 1, 1))], dim=1))

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3_1(e3)
        if self.step_flag <= 2:
            e3 = self.Conv3_2(torch.cat([e3, self.Avgpool(self.Avgpool(param_layer))], dim=1))
        elif self.step_flag > 2:
            e3 = self.Conv3_2(torch.cat([e3, self.Avgpool(self.Avgpool(
                self.param_layer.repeat(x.size(0), 1, 1, 1)))], dim=1))

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4_1(e4)
        if self.step_flag <= 2:
            e4 = self.Conv4_2(torch.cat([e4, self.Avgpool(self.Avgpool(self.Avgpool(param_layer)))], dim=1))
        elif self.step_flag > 2:
            e4 = self.Conv4_2(torch.cat([e4, self.Avgpool(self.Avgpool(self.Avgpool(
                self.param_layer.repeat(x.size(0), 1, 1, 1))))], dim=1))
        
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5_1(e5)
        if self.step_flag <= 2:
            e5 = self.Conv5_2(torch.cat([e5,self.Avgpool(self.Avgpool(self.Avgpool(self.Avgpool(param_layer))))], dim=1))
        elif self.step_flag > 2:
            e5 = self.Conv5_2(torch.cat([e5, self.Avgpool(self.Avgpool(self.Avgpool(self.Avgpool(
                self.param_layer.repeat(x.size(0), 1, 1, 1)))))], dim=1))


        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5_1(d5)
        d5 = self.Up_conv5_2(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4_1(d4)
        d4 = self.Up_conv4_2(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3_1(d3)
        d3 = self.Up_conv3_2(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2_1(d2)
        d2 = self.Up_conv2_2(d2)

        out = self.Conv(d2)

        return out

if __name__ == '__main__':
    net = U_Net(3, 3, step_flag=3, img_size=512)
    # print(net.module.param_layer)