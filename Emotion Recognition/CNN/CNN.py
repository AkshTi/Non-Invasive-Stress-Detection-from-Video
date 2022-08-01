import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

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

transform_train = transforms.Compose([transforms.ToPILImage(), transforms.Resize(48), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(48), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

images_train = np.load(r'/content/drive/MyDrive/Stress Detection/CKs/CK+/train_images.npy', encoding='latin1')
train_labels = np.load(r'/content/drive/MyDrive/Stress Detection/CKs/CK+/train_labels.npy', encoding='latin1')
print(images_train.shape)
images_test = np.load(r'/content/drive/MyDrive/Stress Detection/CKs/CK+/test_images.npy', encoding='latin1')
test_labels = np.load(r'/content/drive/MyDrive/Stress Detection/CKs/CK+/test_labels.npy', encoding='latin1')


for i in range(0, len(images_train)):
    X_train_final.append(torch.Tensor(images_train[i].reshape((1, 227, 227)))/255)
for i in range(0, len(images_test)):
    X_test_final.append(torch.Tensor(images_test[i].reshape((1, 227, 227)))/255)
    
X_train_label = torch.LongTensor(train_labels)
X_test_label = torch.LongTensor(test_labels)

trainset = custom_dataset(X_train_final,X_train_label, transforms=transform_train)
testset = custom_dataset(X_test_final, X_test_label, transforms=transform_test)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_Size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.015, momentum=0.9)

