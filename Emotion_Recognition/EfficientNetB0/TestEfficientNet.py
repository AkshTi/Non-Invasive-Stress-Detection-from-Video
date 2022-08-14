
# Visualization and check of model's performance.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from TrainEfficientNet import *

#Test to read in an image.
model = load_model("path")
image_array = cv2.imread("an image")
plt.imshow(image_array)
 
# Test prediction on single image.
image_array = cv2.resize(image_array, (48, 48))
image_array = np.reshape(image_array, (1, 48, 48, 3))
predict = model.predict(image_array)

mapping = {0: "anger", 1:'contempt', 2: "disgust", 3:'fear', 4:'happy', 5:"neutral", 6:'sadness', 7:"surprise"}

"""neutral is 5
contempt is 1
anger is 0
fear is 3
happy is 4
sadness is 6
surprise is 7
"""
print({mapping[np.argmax(predict)]})
print(np.argmax(predict))
X_test, y_test = yields()

#Test on several images picked randomly.
samples = np.random.choice(len(X_test), 7)
predictions = model.predict(X_test[samples], verbose=0)
fig, axes = plt.subplots(len(samples), 2, figsize=(18, 13))
fig.subplots_adjust(hspace=0.3, wspace=-0.2)

#Visualization.
for i, (prediction, image, label) in enumerate(zip(predictions, X_test[samples], y_test[samples])):
    axes[i, 0].imshow(np.squeeze(image/255.))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(1., -3, f'Actual Emotion: [{mapping[np.argmax(label)]}], [{np.argmax(label)}]', weight='bold')
    axes[i, 1].bar(np.arange(len(prediction)), prediction)
    axes[i, 1].set_xticks(np.arange(len(prediction)))
    axes[i, 1].set_title(f"Model's Prediction: [{mapping[np.argmax(prediction)]}], [{np.argmax(label)}]", weight='bold')