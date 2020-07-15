#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
import sys
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset import ImageDataset
from network import totalnet
from SILoss import SILoss
from SSIMLoss import SSIMLoss
siLoss=SILoss().cuda()
ssimloss=SSIMLoss().cuda()
l1loss=torch.nn.L1Loss().cuda()

trainloader = DataLoader(
    ImageDataset(transforms_=None),
    batch_size=2,
    shuffle=False,drop_last=True
)

net=totalnet()
net.cuda()
net.train()

optimizer_G = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.5, 0.999))
print("data length:",len(trainloader))
print("start training")
eopchnum=200
for epoch in range(0, eopchnum):
  print("epoch:",epoch)
  iteration=0
  for i, total in enumerate(trainloader):
    iteration=iteration+1
    real_img = total["img"]
    real_oral=total["oral"]
    real_grad = total["grad"]
    real_trans=total["trans"]
    real_ref=total["ref"]
    real_img=real_img.cuda()
    real_grad=real_grad.cuda()
    real_oral=real_oral.cuda()
    real_trans=real_trans.cuda()
    real_ref=real_ref.cuda()

    optimizer_G.zero_grad()
    print('t',real_img.shape)
    #with torch.no_grad():
    trans,ref,grad_output=net(real_img,real_oral)
    print('outputgrad',grad_output.shape)
    loss1=siLoss(grad_output,real_grad)
    loss2=ssimloss(trans,real_trans)
    loss3=ssimloss(ref,real_ref)
    loss4=l1loss(trans,real_trans)
    loss=Variable(loss1+0.8*loss2+loss3+loss4,requires_grad=True)
    loss.backward()
    optimizer_G.step()
    print("batch:%3d,iteration:%3d,loss_g:%3f"%(epoch+1,iteration,loss.item()))
  if(epoch%10==0):
    torch.save(net,"Gin_%3d.pth"%epoch)