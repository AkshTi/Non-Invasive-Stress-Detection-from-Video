#CNN model file. 
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

X_train_final = []
X_test_final = []
train_accuracy = []
test_accuracy = []

batch_Size = 30
number_of_epochs = 100 

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
  
        self.conv1 = nn.Conv2d(1, 32, 5, padding=0) 
        self.conv2 = nn.Conv2d(32, 64, 5, padding=0)        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=0)
        self.drop1 = nn.Dropout2d(p=0.3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 9 * 9, 2048)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(1024, 8)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm1d(2048)
        self.bn5 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.drop1(x)
        x = self.pool2(x)
        x = x.view(-1, 128 * 9 * 9)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn5(x)
        x = self.drop3(x)
        x = self.fc3(x)
        return x
    
def weight(labels):
    scale = torch.FloatTensor(8)
    for i in range(8):
        scale[i] = ((labels==i).sum())
    return scale.max() / scale
  
class custom_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):   
        dat = self.data[index]
        if self.transforms is not None:
            dat = self.transforms(dat)
        transform = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])  
        return (dat,self.labels[index])
   
    def __len__(self):
        return len(self.data)



