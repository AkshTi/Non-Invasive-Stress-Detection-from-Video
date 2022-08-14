import torch
import matplotlib.pyplot as plt
from CNN import *

transform_train = transforms.Compose([transforms.ToPILImage(), transforms.Resize(48), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(48), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

#Make loadable data reserves
images_train = np.load(r'/content/drive/MyDrive/Stress Detection/CKs/CK+/train_images.npy', encoding='latin1')
train_labels = np.load(r'/content/drive/MyDrive/Stress Detection/CKs/CK+/train_labels.npy', encoding='latin1')
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

#load model and initialize optimizer.
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.015, momentum=0.9)

#Train
for epoch in range(number_of_epochs):  
    correct = 0.
    total = 0.
    model.train()
    running_loss_train = 0.0
    running_loss_test = 0.0
    for i, data in enumerate(train_loader, 0):       
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()
    train_accuracy.append(correct / total)
    
    correct = 0.
    total = 0.
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            loss = criterion(outputs, labels.to(device))
            print("Loss: ", loss)
    test_accuracy.append(correct/total)

    print('Epoch:', epoch)
    print('Train accuracy:', train_accuracy[epoch]*100)
    print('Validation accuracy:', test_accuracy[epoch]*100)  
    torch.save(model.state_dict(), "/content/drive/MyDrive/Stress Detection/CKs/recog.pth")

#Plot the model files.
plt.figure()
plt.plot(range(number_of_epochs),train_accuracy, 'sk-', label='Train')
plt.plot(range(number_of_epochs),test_accuracy, 'sr-', label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
torch.cuda.empty_cache()
plt.show()
###################
