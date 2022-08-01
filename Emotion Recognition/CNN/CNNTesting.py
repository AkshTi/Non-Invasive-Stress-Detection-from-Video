import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from CNNTraining import *

def getLabel(id):
  print(id)
  return ['neutral', 'anger','contempt','disgust','fear','happy','sadness','surprise'][id]
plt.figure(figsize=(10,10))

index = 59
#with torch.no_grad():
  #for data in list(test_loader)[:10]:

for i in range(0, 9):
  plt.subplot(330+1+i)
  plt.imshow(images[i,0], cmap=plt.get_cmap("gray"))
  plt.gca().get_xaxis().set_ticks([])
  plt.gca().get_yaxis().set_ticks([])
  plt.ylabel("prediction = %s" % getLabel(labels[i]))
plt.show()
plt.savefig("savefigpath")