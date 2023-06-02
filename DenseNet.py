import torch, math
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growrate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growrate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(in_channels=nChannels, out_channels=interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(in_channels=interChannels, out_channels=growrate, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))#torch.Size([1, 24, 112, 112])
        out = self.conv2(F.relu(self.bn2(out)))#torch.Size([1, 12, 112, 112])
        out = torch.cat((out, x), dim=1)#torch.Size([1, 36, 112, 112])
        return out
    
class Transiton(nn.Module):
    def __init__(self, nChannels, outChannels):
        super(Transiton, self).__init__()
        self.bn = nn.BatchNorm2d(nChannels)
        self.conv = nn.Conv2d(in_channels=nChannels, out_channels=outChannels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
    
        return x

class DenseNet(nn.Module):
    def __init__(self, growrate=32, depth=6, reduction=0.5, nClasses=100):
        super(DenseNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=growrate, kernel_size=7, padding=3, stride=2, bias=False)
        self.Max_Pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        nBottlenecks = depth
        nChannels = growrate
        self.dense1 = self._make_dense(nChannels=nChannels, growrate=growrate, nBottleneck=nBottlenecks)#torch.Size([1, 224, 56, 56])

        nChannels += growrate * nBottlenecks #32 + 32 * 6
        outChannels = int(math.floor(nChannels * reduction))#112
        self.transition1 = Transiton(nChannels=nChannels, outChannels=outChannels)#torch.Size([1, 112, 56, 56])
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)#torch.Size([1, 112, 28, 28])

        nChannels = outChannels #112
        nBottlenecks = 2 * nBottlenecks#12
        self.dense2 = self._make_dense(nChannels=nChannels, growrate=growrate, nBottleneck=nBottlenecks)#torch.Size([1, 496, 28, 28])

        nChannels += growrate * nBottlenecks#496
        outChannels = int(math.floor(nChannels * reduction))#248
        self.transition2 = Transiton(nChannels=nChannels, outChannels=outChannels)#torch.Size([1, 248, 28, 28])
        # self.avg_pool  torch.Size([1, 248, 14, 14])

        nChannels = outChannels#248
        nBottlenecks = 2 * nBottlenecks#24
        self.dense3 = self._make_dense(nChannels=nChannels, growrate=growrate, nBottleneck=nBottlenecks)#torch.Size([1, 1016, 14, 14])

        nChannels += nBottlenecks * growrate#1016
        outChannels = int(math.floor(nChannels * reduction))#508
        self.transition3 = Transiton(nChannels=nChannels, outChannels=outChannels)#torch.Size([1, 508, 14, 14])
        # self.avg_pool  torch.Size([1, 508, 7, 7])

        nChannels = outChannels#508
        nBottlenecks = 16
        self.dense4 = self._make_dense(nChannels=nChannels, growrate=growrate, nBottleneck=nBottlenecks)#torch.Size([1, 1020, 7, 7])
        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=7)#torch.Size([1, 1020, 1, 1])

        nChannels += nBottlenecks*growrate
        self.linear = nn.Linear(nChannels, nClasses)
        self.softmax = nn.Softmax()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)#torch.Size([1, 12, 112, 112])
        x = self.Max_Pool(x)
        x = self.dense1(x)
        x = self.transition1(x)
        x = self.avg_pool(x)
        x = self.dense2(x)
        x = self.transition2(x)
        x = self.avg_pool(x)
        x = self.dense3(x)
        x = self.transition3(x)
        x = self.avg_pool(x)
        x = self.dense4(x)
        x = self.global_pool(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        # x = self.softmax(x)
        x = F.log_softmax(x)

        return x
    
    def _make_dense(self, nChannels, growrate, nBottleneck):
        layers = []
        for i in range(int(nBottleneck)):
            layers.append(Bottleneck(nChannels, growrate))
            nChannels += growrate
        return nn.Sequential(*layers)
    
    
# model = DenseNet()
# inputs = torch.randn(((1, 3, 224, 224)))
# output = model(inputs)
# print(output.size())