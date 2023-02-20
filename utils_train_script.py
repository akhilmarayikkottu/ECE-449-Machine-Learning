import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as dset
from hw2_ResNet import ResNet
import struct
import os
import csv

torch.manual_seed(1)


root = './hw2_data'
if not os.path.exists(root):
    os.mkdir(root)

normalization = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=normalization, download=True)
test_set = dset.MNIST(root=root, train=False, transform=normalization, download=True)
trainLoader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
testLoader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

#####################################################################
f = open('Testing_C4', 'w')
#writer = csv.writer(f)
TestC4 = []
TrainingLoss = []


net = ResNet(64)

numparams = 0
for f in net.parameters():
    print(f.size())
    numparams += f.numel()

optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=0)
optimizer.zero_grad()

criterion = nn.CrossEntropyLoss()

def test(net, testLoader):
    net.eval()
    correct = 0
    with torch.no_grad():
        ii = 0
        running_loss = 0.0
        for (data,target) in testLoader:
            output = net(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
            running_loss += loss.item()
            ii  += 1
 
        print("Test Accuracy: %f" % (100.*correct/len(testLoader.dataset)))
        temp =  (100.*correct/len(testLoader.dataset))
        epoch_loss = running_loss /ii
        return(epoch_loss)

#test(net, testLoader)
for epoch in range(400):
    net.train()
    ii = 0
    running_loss = 0.0 
    for batch_idx, (data, target) in enumerate(trainLoader):
        pred = net(data)
        loss = criterion(pred, target)
        loss.backward()
        gn = 0
        for f in net.parameters():
            gn = gn + torch.norm(f.grad)
        #print("E: %d; B: %d; Loss: %f; ||g||: %f" % (epoch, batch_idx, loss, gn))
        running_loss += loss.item() 
        ii  += 1
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss = running_loss /ii
    TrainingLoss.append(epoch_loss)

    
    loss = test(net, testLoader)
    TestC4.append(loss)


plt.plot(TestC4,'r')
plt.xlabel('epoch')
plt.ylabel('testing loss')
plt.show()
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.plot(TrainingLoss,'b')
plt.show()
