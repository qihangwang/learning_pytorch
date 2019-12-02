# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         train_net
# Description:  
# Author:       Administrator
# Date:         2019/12/1
#-------------------------------------------------------------------------------
from torch import optim
import torch as t
import torch.nn as nn
import net_pytorch
from torch.autograd import Variable
import torchvision as tv
import torchvision.transforms as transforms
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#定义网络
net = net_pytorch.Net()
net.to(device)
#定义损失函数和优化器
criterion  = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum=0.9)
#导入数据
transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = tv.datasets.CIFAR10(
    root = "/tmp/data",
    train = True,
    download=False,
    transform = transform
)
trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size = 4,
    shuffle = True,
    num_workers = 2
)
#训练网络
for epoch in range(2):
    running_loss = 0.0
    for i ,data in enumerate(trainloader,0):
        #输入数据
        inputs,labels = data
        inputs,labels = Variable(inputs),Variable(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        #梯度清零
        optimizer.zero_grad()
        #forward+backforward
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        #更新参数
        optimizer.step()
        #打印log信息
        running_loss += loss.item()
        if i%2000 == 1999:#每2000个batch打印一次训练状态
            print('[%d,%5d] loss:%.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0
print('Finish Training')