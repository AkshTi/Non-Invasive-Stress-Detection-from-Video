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
import glob, tempfile
import time, datetime
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
#import dlib
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings("ignore")

def getPicfromVideo(videofilename, framedirectory):
  import cv2
  #length = len([entry for entry in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, entry))])
  print("vid_path", videofilename)
  vidcap = cv2.VideoCapture(videofilename)

  frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

  for i in range(frame_count):
    hasFrames,image = vidcap.read()
    if hasFrames:
        # if xres is larger than yres, remove x boundaries 
        yres, xres, _ = image.shape 
        if xres > yres:
          crop = (xres - yres) // 2
          image = image[:,crop:-crop,:]
        if yres > xres:
          crop = (yres - xres) // 2
          image = image[crop:-crop,:,:]

        path = os.path.join(framedirectory, str(i).zfill(3)+".jpg")
        cv2.imwrite(path, image)     # save frame as JPG file

  # def getFrame(sec):
  #     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
  #     hasFrames,image = vidcap.read()
  #     if hasFrames:
  #         path = os.path.join(framedirectory, "image"+str(count)+".jpg")
  #         cv2.imwrite(path, image)     # save frame as JPG file
  #     return hasFrames

  # sec = 0
  # framespersecond = vidcap.get(cv2.CAP_PROP_FPS)
  # frameRate = 1/framespersecond #//it will capture image in each 0.5 second
  # count=1
  # success = getFrame(sec)
  # while success:
  #     count = count + 1
  #     sec = sec + frameRate
  #     sec = round(sec, 2)
  #     success = getFrame(sec)
  # #length = len([entry for entry in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, entry))])
  # # get length, fps, and framecount of the video

def getMetrics(videofilename):
  cap = cv2.VideoCapture(videofilename)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration = frame_count // fps
  return duration, fps, frame_count


def createOutputFolders(OUTDIR):
  USERDIR = tempfile.mkdtemp(prefix=str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))+'___', dir=OUTDIR)

  FRAMESDIR = os.path.join(USERDIR, 'frames')
  os.makedirs(FRAMESDIR, exist_ok=True)

  PLOTSDIR = os.path.join(USERDIR, 'plots')
  os.makedirs(PLOTSDIR, exist_ok=True)

  return USERDIR, FRAMESDIR, PLOTSDIR

