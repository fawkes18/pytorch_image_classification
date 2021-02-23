import torch.nn as nn
import torch
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #color channels = 1, output_size = 32, kernel_size (3x3)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) #input size needs to be equal to last output size
        self.fc1 = nn.Linear(4096, 256) #img_size = (32x32) -> max_pool1 = (16x16) ->max_pool2 = (8x8) [output_size*length*width]
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.pool = nn.MaxPool2d(2, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4096)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x







