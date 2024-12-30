import torch
from torch import nn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.onnx

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 62)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv3_bn(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(self.conv4_bn(x))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = F.softmax(self.fc2(x), -1)
        return x

print("Hello!")
model_path = 'parkingpytorchmodel.pt'
torch_model = CustomCNN()
torch_model.load_state_dict(torch.load(model_path).module.state_dict())
torch_model.eval()

print("Hello!!")

dummy_input = torch.randn(1, 3, 150, 150)
torch.onnx.export(torch_model, dummy_input, "parkingpytorchmodel.onnx")
