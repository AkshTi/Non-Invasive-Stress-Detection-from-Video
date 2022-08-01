import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

for epoch in range(number_of_epochs):  
    correct = 0.
    total = 0.
    
    # if epoch> 10:
    #optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.015, momentum=0.9)
    # elif epoch > 50:
    #   optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.00015, momentum=0.9)
    
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
###################################################
plt.figure()
plt.plot(range(number_of_epochs),train_accuracy, 'sk-', label='Train')
plt.plot(range(number_of_epochs),test_accuracy, 'sr-', label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
torch.cuda.empty_cache()
plt.show()
###################
