from scipy.signal import butter, lfilter
import torch
from torch.utils.data import Sampler
import glob
import os
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from scipy.signal import resample
import math
import argparse
import glob
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import json
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import heartpy as hp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import imghdr
from PIL import Image
import itertools
import seaborn as sns
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import matplotlib.image as mpimg
import warnings
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
warnings.filterwarnings("ignore")
from Common import *

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = ConvNet().to(device)

def getSizes(directory):
  for file in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, file)):
      f_img = os.path.join(directory, file)
      img = Image.open(f_img)
      img = img.resize((227,227))
      img.save(f_img)

def getEmotions(directory):
  mapping = {"anger":0, 'contempt':1, "disgust":2, 'fear':3, 'happy':4, "neutral":5, 'sadness':6, "surprise":7}
  #mapping = {0: "anger", 1:'neutral', 2: "disgust", 3:'fear', 4:'happy', 5:"sadness", 6:'surprise'}
  data = []
  for filename in sorted(os.listdir(directory)):
    if os.path.isfile(os.path.join(directory, filename)):
      image = cv2.imread(os.path.join(directory, filename))
      image = cv2.resize(image, (48, 48))
      data.append(image)
  model = load_model("weights.hdf5")
  predictionList = []
  for image in data:
    image = image.reshape(1, 48, 48, 3)
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    predictionList.append(prediction)
  return predictionList

def getMax(emotion_count, emotion_list):
  sns.color_palette("husl", 8)
  sns.barplot(emotion_list, emotion_count, palette = "husl")
  plt.xticks(rotation=15)

def getTimeList(emotions):
  length = 0
  emotion_lists = []
  for i in range(len(emotions)):
    emotion_lists.append(emotions[i])
  return emotion_lists, len(emotions)

def timeTrends(emotion_list, emotions, duration, PLOTSDIR):
  #duration, fps, frame_count = getMetrics(os.path.join())
  emotion_lists, length = getTimeList(emotions)
  times = list(range(0, length))
  xranges = np.linspace(0, duration, length)
  sns.scatterplot(x=xranges, y=emotion_lists, palette = "Set2")
  plt.title("Emotion Detection (By Frame)")
  plt.yticks(range(0, len(emotion_list)), emotion_list)
  plt.xlabel("Time (seconds)")
  plt.xticks(rotation=15)
  plt.savefig(os.path.join(PLOTSDIR, 'stage1_emotions.png'))
  plt.clf()

def getEmfromVideo(filepath, duration, PLOTSDIR):
  #emotion_list = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]
  emotion_list= ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]
  emotion_count = [0, 0, 0, 0, 0, 0, 0, 0]
  emotions = getEmotions(filepath)
  for i in range(len(emotions)):
    int_tensor = emotions[i]
    emotion_count[int_tensor] +=1
  emotion_lists, length = getTimeList(emotions)
  timeTrends(emotion_list, emotions, duration, PLOTSDIR)
  return emotion_lists


