import os
import cv2
import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
cap = cv2.VideoCapture(r"C:\Users\Pradeep.Tiwari\Non-Invasive-Stress-Detection-from-Video\tmpzo5jc6vq.mp4")
# print(cap.get(cv2.CAP_PROP_FPS))

def createfolder(videopath, videoname, width, height):
  new_dir = os.path.join(videoname[:-4] + "resize")
#   new_dir = os.path.join(videopaths, videoname[:-4] + "resize")
  print("new_dir: ", new_dir)
  if not os.path.exists(new_dir):
    os.makedirs(new_dir)
  cap = cv2.VideoCapture(videopath)
  total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  count = 0
  while count<total_frames:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (height, width), interpolation = cv2.INTER_AREA)
    #print(frame.shape)
    cv2.imwrite(os.path.join(new_dir, "i"+str(count)+".png"), frame)
    count+=1
  return new_dir, cap.get(cv2.CAP_PROP_FPS), len(os.listdir(new_dir))

createfolder(r"C:\Users\Pradeep.Tiwari\Non-Invasive-Stress-Detection-from-Video\tmpzo5jc6vq.mp4", "tmpzo5jc6vq.mp4", 320, 240)

def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# Helper Methods
def getHR(instance):
  videoFrameRate = 29.583333333333332
  # Webcam Parameters
  realWidth = 320
  realHeight = 240
  videoWidth = 160
  videoHeight = 120
  videoChannels = 3

  #print(f"Metrics: Height: {videoHeight}, FPS {videoFrameRate}, Width: {videoWidth}")
  # Color Magnification Parameters
  levels = 3
  alpha = 170
  minFrequency = 1.0
  maxFrequency = 1.8
  bufferSize = 150
  bufferIndex = 0

  # Initialize Gaussian Pyramid
  firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
  firstGauss = buildGauss(firstFrame, levels+1)[levels]
  videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
  fourierTransformAvg = np.zeros((bufferSize))

  # Bandpass Filter for Specified Frequencies
  frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
  mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

  # Heart Rate Calculation Variables
  bpmCalculationFrequency = 15
  bpmBufferIndex = 0
  bpmBufferSize = 10
  bpmBuffer = np.zeros((bpmBufferSize))

  i = 0
  bpm_list = []


  for file in os.listdir(instance):
      frame = cv2.imread(os.path.join(instance, file))     
      detectionFrame = frame[videoHeight//2:realHeight-videoHeight//2, videoWidth//2:realWidth-videoWidth//2, :]

      # Construct Gaussian Pyramid
      videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
      fourierTransform = np.fft.fft(videoGauss, axis=0)

      # Bandpass Filter
      fourierTransform[mask == False] = 0

      # Grab a Pulse
      if bufferIndex % bpmCalculationFrequency == 0:
          i = i + 1
          for buf in range(bufferSize):
              fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
          hz = frequencies[np.argmax(fourierTransformAvg)]
          bpm = 60.0 * hz
          bpmBuffer[bpmBufferIndex] = bpm
          bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

      # Amplify
      filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
      filtered = filtered * alpha
      bufferIndex = (bufferIndex + 1) % bufferSize
      
      if i > bpmBufferSize:
          bpm_list.append(bpmBuffer.mean())
      else:
          bpm_list.append(0)

  #generate a time in seconds
  bpm_list_actual = [i for i in bpm_list if i is not 0]
  time = len(bpm_list_actual)//videoFrameRate
  num_zeroes = len(bpm_list)-len(bpm_list_actual)
  empty_seconds= num_zeroes//videoFrameRate
  x = np.linspace(empty_seconds, time+empty_seconds, len(bpm_list_actual))
  plt.plot(x, bpm_list_actual)
  #print("avg: ", sum(bpm_list_actual)/len(bpm_list_actual))
  plt.show()
  print(bpm_list_actual)
  return bpm_list_actual

#bpm_list_actual = getHR("/content/drive/MyDrive/Stress Detection/0801video.avi", "0801video.avi")
#bpm_list_actual = getHR("/content/drive/MyDrive/Stress Detection/VIDEOS/Copy of Test1.mp4", "Copy of Test1.mp4")
#bpm_list_actual = getHR("/content/drive/MyDrive/Stress Detection/VIDEOS/Copy of Test4.mp4", "Copy of Test4.mp4")
#bpm_list_actual = getHR("/content/drive/MyDrive/Stress Detection/0805video.avi", "0805video.avi")
#bpm_list_actual = getHR("/content/drive/MyDrive/Stress Detection/video.avi", "video.avi")
bpm_list_actual = getHR(r"C:\Users\Pradeep.Tiwari\Non-Invasive-Stress-Detection-from-Video\webapp\tmpzo5jc6vqresize")