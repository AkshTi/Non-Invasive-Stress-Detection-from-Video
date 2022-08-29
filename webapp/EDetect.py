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
import dlib
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings("ignore")
from Common import *
global points, points_lip, emotion_classifier

def ebdist(leye,reye):
    eyedist = dist.euclidean(leye,reye)
    points.append(int(eyedist))
    return eyedist

def lpdist(l_lower,l_upper):

    lipdist = dist.euclidean(l_lower[0], l_upper[0])
    points_lip.append(int(lipdist))
    return lipdist
    
def normalize_values(points,disp,points_lip,dis_lip):
    normalize_value_lip = abs(dis_lip - np.min(points_lip))/abs(np.max(points_lip) - np.min(points_lip))
    normalized_value_eye =abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    normalized_value =(normalized_value_eye + normalize_value_lip)/2
    stress_value = (np.exp(-(normalized_value)))
    if stress_value>=0.75:
        stress_label="High Stress"
    else:
        stress_label="Low Stress"
    return stress_value,stress_label
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"landmarks.dat")
points = []
points_lip = []
def get_frame(directory, duration, fps, frame_count): 
    stress_value_list = []
    stress_level_list = []
    cap = cv2.VideoCapture(directory)
    count = 0
    e = 0
    prev_hash = 0
    chooseframeseachsecond = 5
    while cap.isOpened():
        if (e%chooseframeseachsecond==0):
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            ret,frame = cap.read()
            frame = cv2.flip(frame,1)

            # if xres is larger than yres, remove x boundaries 
            yres, xres, _ = frame.shape 
            if xres > yres:
              crop = (xres - yres) // 2
              frame = frame[:,crop:-crop,:]
            if yres > xres:
              crop = (yres - xres) // 2
              frame = frame[crop:-crop,:,:]

            if frame is not None:
              frame = imutils.resize(frame, width=500,height=500)

              (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
              (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
               # lip aka mouth
              (l_lower, l_upper) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

              #preprocessing the image
              gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
              count +=1
              detections = detector(gray,0)
              for detection in detections:
                shape = predictor(frame,detection)
                shape = face_utils.shape_to_np(shape)

                leyebrow = shape[lBegin:lEnd]
                reyebrow = shape[rBegin:rEnd]
                openmouth = shape[l_lower:l_upper]

                reyebrowhull = cv2.convexHull(reyebrow)
                leyebrowhull = cv2.convexHull(leyebrow)
                openmouthhull = cv2.convexHull(openmouth) # figuring out convex shape when lips opened
                # Measuring lip aka "open mouth" and eye distance
                lipdist = lpdist(openmouthhull[-1],openmouthhull[0])
                eyedist = ebdist(leyebrow[-1],reyebrow[0])

                stress_value,stress_label = normalize_values(points,eyedist, points_lip, lipdist)
                if stress_value>0.85:
                    stress_value = 0.85
                stress_value_list.append(stress_value)
                prev_hash = stress_value
                stress_level_list.append(stress_label)

                if count==frame_count:
                  cap.release()
        else:
            stress_level_list.append(prev_hash)
        e+=1
    return stress_value_list, stress_level_list, fps, frame_count
