import torchvision
import torch.nn as nn
import torch


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        resnet50 = torchvision.models.resnet34(pretrained=False)
        self.base= nn.Sequential(*list(resnet50.children())[:-1])
        N=512
        self.fc1 = nn.Linear(in_features=N, out_features=7, bias=False)
        self.fc2 = nn.Linear(in_features=N, out_features=12, bias=False)
        
    def forward(self,x):
        x = self.base(x)
        f = x.view(x.size(0),-1)
        return [self.fc1(f),self.fc2(f)]


    
    
