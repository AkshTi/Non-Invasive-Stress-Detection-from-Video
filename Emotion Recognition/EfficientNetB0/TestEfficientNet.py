
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dropout, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, plot_model
from keras.models import load_model
from TrainEfficientNet import *

model = load_model("path")
image_array = cv2.imread("an image")
plt.imshow(image_array)
 
image_array = cv2.resize(image_array, (48, 48))
image_array = np.reshape(image_array, (1, 48, 48, 3))
predict = model.predict(image_array)
#mapping = {0: "anger", 1:'neutral', 2: "disgust", 3:'fear', 4:'happy', 5:"sadness", 6:'surprise'}
mapping = {0: "anger", 1:'contempt', 2: "disgust", 3:'fear', 4:'happy', 5:"neutral", 6:'sadness', 7:"surprise"}
#mapping = {0: "happy", 1:'anger', 2: "contempt", 3:'disgust', 4:'happy', 5:'neutral', 6:'sadness', 7:'surprise'}
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
samples = np.random.choice(len(X_test), 7)

predictions = model.predict(X_test[samples], verbose=0)

fig, axes = plt.subplots(len(samples), 2, figsize=(18, 13))
fig.subplots_adjust(hspace=0.3, wspace=-0.2)

for i, (prediction, image, label) in enumerate(zip(predictions, X_test[samples], y_test[samples])):

    axes[i, 0].imshow(np.squeeze(image/255.))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(1., -3, f'Actual Emotion: [{mapping[np.argmax(label)]}], [{np.argmax(label)}]', weight='bold')

    axes[i, 1].bar(np.arange(len(prediction)), prediction)
    axes[i, 1].set_xticks(np.arange(len(prediction)))
    axes[i, 1].set_title(f"Model's Prediction: [{mapping[np.argmax(prediction)]}], [{np.argmax(label)}]", weight='bold')