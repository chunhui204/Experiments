import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicBlock(nn.Module):
  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride>1 or in_planes != planes:
      self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                                     nn.BatchNorm2d(planes))


  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)

    out = F.relu(self.bn2(self.conv2(out)))
    x = self.shortcut(x)
    return F.relu(x + out)

class Resnet(nn.Module):
  def __init__(self, num_layer, num_class):
    super(Resnet, self).__init__()
    n = int((num_layer - 2)/ 6)
    self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)

    self.layer1 = self._make_layer(n, 16, 16, 1)
    self.layer2 = self._make_layer(n, 16, 32, 2)
    self.layer3 = self._make_layer(n, 32, 64, 2)
    self.fc = nn.Linear(64, num_class)

  def _make_layer(self, num_block, in_planes, planes, stride):
    layers = []
    for i in range(num_block):
      if  i== 0:
        layers.append(BasicBlock(in_planes, planes, stride))
      else:
        layers.append(BasicBlock(planes, planes, 1))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = F.avg_pool2d(x, (8,8))
    x = x.view(x.size(0),-1)
    x = self.fc(x)

    return F.log_softmax(x, 1)

def Resnet56():
  return Resnet(56, 10)
