#Visualize output.

import numpy as np
import matplotlib.pyplot as plt
from CNNTraining import *

#returns label of encoding.
def getLabel(id):
  print(id)
  return ['neutral', 'anger','contempt','disgust','fear','happy','sadness','surprise'][id]
plt.figure(figsize=(10,10))

index = 59

for i in range(0, 9):
  plt.subplot(330+1+i)
  plt.imshow(images[i,0], cmap=plt.get_cmap("gray"))
  plt.gca().get_xaxis().set_ticks([])
  plt.gca().get_yaxis().set_ticks([])
  plt.ylabel("prediction = %s" % getLabel(labels[i]))
plt.show()
plt.savefig("savefigpath")