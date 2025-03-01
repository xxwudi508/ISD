#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset1
from net1  import MAEFNet
import time

class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot='./out/swin/modelTest', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.net.load_state_dict(torch.load(self.cfg.snapshot))

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image)
                out = out5r

                plt.figure()
                plt.axis('off')
                plt.imshow(np.uint8(image[0].permute(1,2,0).cpu().numpy()*self.cfg.std + self.cfg.mean))

                plt.figure()
                plt.axis('off')
                plt.imshow(mask[0].cpu().numpy())
                #
                plt.figure()
                plt.axis('off')

                plt.imshow(out2u[0, 0].cpu().numpy(),cmap='gray')

                plt.ioff()
                plt.show()

    
    def save(self):

        with torch.no_grad():
            total_time = 0

            for image, mask, shape, name in self.loader:
                image = image.cuda().float()

                start_time = time.time()
                _, out2u, _, _, _, _ = self.net(image, shape)
                torch.cuda.synchronize()
                end_time = time.time()
                total_time += end_time - start_time

                pred  = (torch.sigmoid(out2u[0,0])*255).cpu().numpy()
                # pred = np.squeeze(out2u[0,0]).cpu().data.numpy()
                head  = './data/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png',  np.round(pred))
                # print(name[0]+'.png')
            print('Total time：{}'.format(total_time))

    def hook(self):
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()

            return hook

        for image, _, shape, name in self.loader:
            image, _ = image.cuda().float(), _.cuda().float()
        # 注册前向钩子
            self.net.res.register_forward_hook(get_activation('res'))
            self.net.msca1.register_forward_hook(get_activation('msca1'))

            # 输入数据
            # input_image = torch.randn(1, 384, 384)  # 示例输入，1张28x28的灰度图

            # 前向传递
            output = self.net(image)

            conv1_activations = activations['res']

            # 绘制 conv1 层的第一个通道
            # plt.figure(figsize=(128, 128))
            # for i in range(conv1_activations.size(1)):
            for i in range(6):

                plt.figure(figsize=(64,64))

                plt.imshow(conv1_activations[0, i].cpu().numpy(), cmap='viridis')

                plt.axis('off')


            plt.show()

            # # 可视化 conv2 层的输出
            conv2_activations = activations['msca1']

            # 绘制 conv2 层的第一个通道
            # plt.figure(figsize=(128, 128))
            for i in range(6):
                plt.figure(figsize=(64,64))
                plt.imshow(conv2_activations[0, i].cpu().numpy(), cmap='viridis')
                plt.axis('off')

            plt.show()

if __name__=='__main__':
    root = './data/'

    for path in ['test']:
        t = Test(dataset1, MAEFNet, root + path)
        t.save()
        # t.show()
        # t.hook()
