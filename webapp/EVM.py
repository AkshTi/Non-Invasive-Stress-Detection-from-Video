import os
import cv2
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

# Helper Methods
def getHR(instance, PLOTSDIR):

  # Webcam Parameters
  realWidth = 320
  realHeight = 240
  videoWidth = 160
  videoHeight = 120
  videoChannels = 3
  videoFrameRate = 30

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
  cropimage = []
  for x in os.listdir(instance):
    frame = cv2.imread(os.path.join(instance, x))
    cropimage.append(cv2.resize(frame, (320, 240), interpolation = cv2.INTER_AREA))
                     
  for frame in cropimage:    
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
  prepend = []
  num_zeroes = len(bpm_list)-len(bpm_list_actual)
  for i in range(num_zeroes):
        if i >= len(bpm_list_actual):
            i%=bpm_list_actual
        prepend.append(bpm_list_actual[i])
#   prepend = [bpm_list_actual[i] for i in range(num_zeroes)]
  bpm_list_actual = prepend + bpm_list_actual
  time = len(bpm_list_actual)//videoFrameRate
  x = np.linspace(0, time, len(bpm_list_actual))
  plt.plot(x, bpm_list_actual)
  #plt.show()
  print(bpm_list_actual)
  plt.savefig(os.path.join(PLOTSDIR, 'final_stressgraph.png'))
  plt.clf()
  return bpm_list_actual
