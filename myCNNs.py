#%%
#===================================================================
#ECE 5415 - CNN Model Implemented In PyTorch
#(c) 2019  bjy26@cornell.edu, ok93@cornell.edu
#===================================================================

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.autograd import Variable

class musicCNNDeep(nn.Module):

    def __init__(self):
        super(musicCNNDeep, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))

        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 16 * 15, 512)
        self.fc2 = nn.Linear(512, 10)



    def forward(self, x):

        out = self.relu(self.conv1(x))
        out =  self.maxpool(out)
        out = self.drop1(out)

        out = self.relu(self.conv2(out))
        out =  self.maxpool(out)
        out = self.drop1(out)

        out = self.relu(self.conv3(out))
        out =  self.maxpool(out)
        out = self.drop1(out)
    
        out = out.flatten(start_dim=1)

        out = self.relu(self.fc1(out))

        out = self.drop2(out)

        out = self.fc2(out)

        return out

# %%
