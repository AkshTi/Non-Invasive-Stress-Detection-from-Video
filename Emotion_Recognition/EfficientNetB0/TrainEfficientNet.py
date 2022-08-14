# Train and determine accuracy of model.
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dropout, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


# Auxiliary function for data distribution visualization
def data_vis(classes, data):

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(classes, data)
    ax.set(title="Dataset's distribution")
    ax.set(xlabel="Emotions", ylabel="#Images");
    ax.grid()

#load CK+ data from indicated path.
def load_data(data_path):
    subfolders_ck = os.listdir(data_path)
    img_data_list=[]
    labels_list = []
    num_images_per_class = []

    for category in subfolders_ck:
        img_list=os.listdir(data_path +'/'+ category)
        for img in img_list:
            # Load an image from this path
            pixels=cv2.imread(data_path + '/'+ category + '/'+ img )
            face_array=cv2.resize(pixels, None, fx=1, fy=1,interpolation = cv2.INTER_CUBIC)
        
            img_data_list.append(face_array)          
            labels_list.append(category)
            if "neutral" in labels_list:
              print("Neutral")
        
        num_images_per_class.append(len(img_list))
    img_data_list = np.asarray(img_data_list, dtype = "float32", order=None)
    le = LabelEncoder()
    labels = le.fit_transform(labels_list)
    labels = to_categorical(labels, 8)
    print(img_data_list.shape)
    data_vis(subfolders_ck, num_images_per_class)
    return img_data_list, labels

my_path = "my_path_to_the_file"
    
data_path_ck = my_path
data, labels = load_data(data_path_ck)

#splits the testing and trainnig data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, random_state=3)

print(f"X_train has shape: {X_train.shape}")
print(f"y_train has shape: {y_train.shape}\n")
# Helper function for future use.
def yields():
    return X_test, y_test
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=3)

print(f"X_valid has shape: {X_valid.shape}")
print(f"y_valid has shape: {y_valid.shape}\n")
print(f"X_test has shape: {X_test.shape}")
print(f"y_test has shape: {y_test.shape}\n")

print(f"X_train + X_valid + X_test = {X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]} samples in total")
mapping = {0: "anger", 1:'sadness', 2: "disgust", 3:'fear', 4:'happy', 5:"neutral", 6:'contempt', 7:"surprise"}

trainAug = ImageDataGenerator(rotation_range=15,
                              zoom_range=0.15,
                              #width_shift_range=0.2,
                              brightness_range=(.6, 1.2),
                              shear_range=.15,
                              #height_shift_range=0.2,
                              horizontal_flip=True,
                              fill_mode="nearest")

#Build the model, with EfficientNetB0 and other layers.
def build_model():
    inputs = Input(shape=(48, 48, 3))
    base_model = EfficientNetB0(include_top=False, weights='imagenet',
                                drop_connect_rate=0.33, input_tensor=inputs)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(.5, name="top_dropout")(x)
    outputs = Dense(8, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = build_model()

EPOCHS = 100
batch_size = 64
filepath = "FINLAweights.best.hdf5"

# Define the appopriate callbacks.
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
earlystopping = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, mode='auto', restore_best_weights=True)
rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

callbacks = [checkpoint, earlystopping, rlrop]
X_train = np.array([np.array(val) for val in X_train])
y_train = np.array([np.array(val) for val in y_train])
X_valid = np.array([np.array(val) for val in X_valid])
y_valid = np.array([np.array(val) for val in y_valid])

print(X_train[0].shape)
print(y_train.dtype)
print(X_valid.dtype)
print(y_valid.dtype)
hist = model.fit(trainAug.flow(X_train, y_train, batch_size=batch_size),
                 steps_per_epoch=len(X_train) // batch_size,
                 validation_data=(X_valid, y_valid), epochs=EPOCHS, callbacks=callbacks)