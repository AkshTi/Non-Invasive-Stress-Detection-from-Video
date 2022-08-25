from scipy.signal import butter, lfilter
import torch
import os
import torch
import numpy as np
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import seaborn as sns
import warnings
from keras.models import load_model
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from torch.autograd import Variable
from Common import *

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
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
  def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

  predictionList_num = []
  cut_size = 44
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
  ])

  predictionList_name = []
  class_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
  data = []
  predictionList_num = []
  for filename in sorted(os.listdir(directory)):
    path = os.path.join(directory, filename)
    if os.path.isfile(path):
        raw_img = Image.open(path)
        gray = rgb2gray(raw_img)
        gray = raw_img.resize((48,48))

        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)
        ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        data.append(inputs)
        
#       image = cv2.imread(os.path.join(directory, filename))
#       image = cv2.resize(image, (48, 48))
#       data.append(image)

  net = VGG('VGG19')
  checkpoint = torch.load("PrivateTest_model.t7", map_location = device)
  net.load_state_dict(checkpoint['net'])
  net.cuda()
  net.eval()
  model = load_model("weights.hdf5")
  predictionList = []
    
  for image in data:
    outputs = net(inputs)
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)
    predictionList_num.append(int(predicted.cpu().numpy()))
    predictionList_name.append(class_names[int(predicted.cpu().numpy())])
  return predictionList_num, predictionList_name

def getMax(emotion_count, emotion_list):
  sns.color_palette("husl", 8)
  sns.barplot(emotion_list, emotion_count, palette = "husl")
  plt.xticks(rotation=15)

def timeTrends(emotion_list, emotions, duration, PLOTSDIR):
  #duration, fps, frame_count = getMetrics(os.path.join())
  times = list(range(0, len(emotions)))
  xranges = np.linspace(0, duration, len(emotions))
  sns.scatterplot(x=xranges, y=emotions, palette = "Set2")
  plt.title("Emotion Detection (By Frame)")
  #plt.yticks(range(0, len(emotion_list)), emotion_list)
  plt.yticks([0, 1, 2, 3, 4, 5, 6], emotion_list)
  plt.xlabel("Time (seconds)")
  plt.xticks(rotation=15)
  plt.savefig(os.path.join(PLOTSDIR, 'stage1_emotions.png'))
  plt.clf()

def getEmfromVideo(filepath, duration, PLOTSDIR):
  #emotion_list = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]
  emotion_list= ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
  emotion_count = [0, 0, 0, 0, 0, 0, 0]
  emotions, emotion_lists = getEmotions(filepath)
  print(emotions)
  #emotion_lists, length = getTimeList(emotions)
  timeTrends(emotion_list, emotions, duration, PLOTSDIR)
  return emotion_lists


