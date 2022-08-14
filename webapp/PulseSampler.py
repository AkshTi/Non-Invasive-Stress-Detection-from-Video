#-------------imports
from scipy.signal import butter, lfilter
from Common import *
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
warnings.filterwarnings("ignore")

#------------utils

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def psnr(img, img_g):
    criterionMSE = nn.MSELoss()  # .to(device)
    mse = criterionMSE(img, img_g)

    psnr = 10 * torch.log10(torch.tensor(1) / mse)  # 20 *
    return psnr

#---------------Pulse Dataset
device = torch.device('cpu')

class PulseDataset(Dataset):
    """
    PURE, VIPL-hr, optospare and pff pulse dataset. Containing video frames and corresponding to them pulse signal.
    Frames are put in 4D tensor with size [c x d x w x h]
    """

    def __init__(self, sequence_list, root_dir, length, img_height=120, img_width=120, seq_len=1, transform=None):
        """
        Initialize dataset
        :param sequence_list: list of sequences in dataset
        :param root_dir: directory containing sequences folders
        :param length: number of possible sequences
        :param img_height: height of the image
        :param img_width: width of the image
        :param seq_len: length of generated sequence
        :param transform: transforms to apply to data
        """
        seq_list = [sequence_list]
        length = len([entry for entry in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, entry))])
        self.frames_list = pd.DataFrame()
        #adding references (which does notn really neeed)
        for s in seq_list:
            fr_list = glob.glob(root_dir.rstrip() + '\*.jpg')
            
            ref = [i for i in range(len(fr_list))]
            ref = np.array(ref)
            self.frames_list = self.frames_list.append(pd.DataFrame({'frames': fr_list, 'labels': ref}))

        self.length = length
        self.seq_len = seq_len
        self.img_height = img_height
        self.img_width = img_width
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
      
        # if torch.is_tensor(int(idx)):
        #     idx = idx.tolist()
        labels = []
        frames = []
        idxc = idx[-7:-4]
        
        if idxc[0].isdigit():
          idxc = int(idxc)
        elif idx[1].isdigit():
          idxc = idx[1:]
          idxc = int(idxc)
        else:
          idxc = idxc[-1]
          idxc = int(idxc)
        for fr in range(idxc, idxc+self.seq_len):  # frames from idx to idx+seq_len
          if (idxc + self.seq_len <= 255):
            img_name = idx  # path to image
            image = Image.open(img_name)
            image = image.resize((self.img_width, self.img_height))

            if self.transform:
                image = self.transform(image)
            frames.append(image)

        if (len(frames) > 0):
          frames = torch.stack(frames)

          frames = frames.permute(1, 0, 2, 3)
          frames = torch.squeeze(frames, dim=1)
          frames = (frames-torch.mean(frames))/torch.std(frames)*255
          lab = np.array(self.frames_list.iloc[idxc:idxc + self.seq_len, 1])
          labels = torch.tensor(lab, dtype=torch.float)

        sample = (frames, labels)
        return sample

#--------PhysNets

class PhysNet(nn.Module):
    """
    PhysNet with 3D convolution model
    """
    def __init__(self, frames=128):
        """
        Initialise PhysNet model
        :param frames: length of sequence to process
        """
        super(PhysNet, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        # self.attention = SelfAttention(64)
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))  # selects one from every frame of input

    def forward(self, x):  # x [3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]

        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x = self.ConvBlock3(x)  # x [32, T, 64,64]

        x = self.MaxpoolSpaTem(x)  # x [32, T/2, 32,32]    Temporal halve

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x = self.ConvBlock5(x)  # x [64, T/2, 32,32]

        x = self.MaxpoolSpaTem(x)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]

        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]
        x = self.ConvBlock8(F.dropout(x, p=0.2))  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(F.dropout(x, p=0.2))  # x [64, T/4, 8, 8]

        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]
        # h = x.register_hook(self.activations_hook)

        x = self.poolspa(x)  # x [64, T, 1, 1]
        x = self.ConvBlock10(F.dropout(x, p=0.5))  # x [1, T, 1,1]
        #print(x.size(), length)
        rPPG = x.view(-1, length)
        #print(rPPG.size())
        return rPPG, x_visual, x, x_visual1616

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x = self.ConvBlock3(x)  # x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x)  # x [32, T/2, 32,32]    Temporal halve

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        return x

class SelfAttention(nn.Module):
    def __init__(self, c, reduction_ratio=16):
        super(SelfAttention, self).__init__()
        self.decoded = nn.Conv3d(c,  math.ceil(c / reduction_ratio), kernel_size=1)
        self.encoded = nn.Conv3d(math.ceil(c / reduction_ratio), c, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        N = x.size()[0]
        C = x.size()[1]

        decoded = self.decoded(x)
        encoded = self.encoded(self.relu(torch.layer_norm(decoded, decoded.size()[1:])))
        encoded = nn.functional.softmax(encoded)
        cnn = x * encoded
        return cnn

class NegPearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(NegPearson, self).__init__()
        return

    def forward(self, preds, labels):  # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss

# resume = r"C:\Dev\Tools\Python\condabin\stressdetection\Code\s_Drop_3d_32_14.tar"
# print("initialize model...")
# map_location="cpu"
# seq_len = 32 
# model = PhysNet(seq_len)
# model = torch.nn.DataParallel(model)
# #model.cuda()
# ss = sum(p.numel() for p in model.parameters())
# print('num params: ', ss)
# if os.path.isfile(resume):
#     print("=> loading checkpoint '{}'".format(resume))
#     checkpoint = torch.load(resume, map_location=map_location)
#     best_prec1 = checkpoint['best_prec1']
#     model.load_state_dict(checkpoint['state_dict'])
#     print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
# else:
#     print("=> no checkpoint found at '{}'".format(resume)) set CUDA_VISIBLE_DEVICES=""

#-----others
def remove_outliers(listssss):
  m = 1
  u = np.mean(listssss)
  s = np.std(listssss)
  filters = [x for x in listssss if (u - 1 * s < x < u + 1 * s)]
  return filters, s, u

def getVidLength(path):
  cap = cv2.VideoCapture(path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  return frame_count//fps    

#Method
def getHeartRate(path, duration):
    import warnings
    print("\nCALCULATING HEART RATES")
    resume = r"s_Drop_3d_32_14.tar"
    print("\tInitializing model...")
    seq_len = 32 
    model = PhysNet(seq_len)
    model = torch.nn.DataParallel(model)
    ss = sum(p.numel() for p in model.parameters())
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
    end_indexes_test = []
    print(path)
    fr_list = glob.glob(path + os.sep + '*.jpg')
    print("fr_list: ", fr_list)
    end_indexes_test.append(len(fr_list))
    end_indexes_test = [0, *end_indexes_test]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    pulse_test = PulseDataset("", path, seq_len=seq_len,
                                            length=len(fr_list), transform=transforms.Compose([transforms.ToTensor(), normalize]))
    val_loader = torch.utils.data.DataLoader(pulse_test, batch_size=1, shuffle=False, sampler=fr_list, pin_memory=True)
    print("\tRunning model.....")
    model.eval()
    outputs = []
    count = 0
    for k, (net_input, target) in enumerate(val_loader):
        if (isinstance(net_input, list) is not True and isinstance(target, list) is not True):
            with torch.no_grad():
                output, x_visual, x, t = model(net_input)
                outputs.append(output[0])
                count+=1
                #print("count: " + str(count) + "/" + str(frame_count))
    # for k, (net_input, target) in enumerate(val_loader):
    #     if (isinstance(net_input, list) is not True and isinstance(target, list) is not True):
    #         net_input = net_input.cpu(non_blocking=True)
    #         target = target.cpu(non_blocking=True)
    #         with torch.no_grad():
    #             print(net_input.shape)
    #             output, x_visual, x, t = model(net_input)
    #             outputs.append(output[0])
    #             count+=1
    outputs = torch.cat(outputs)
    outputs = (outputs - torch.mean(outputs)) / torch.std(outputs)
    outputs = outputs.tolist()
    fs = 30
    lowcut = 1
    highcut = 3
    yr = butter_bandpass_filter(outputs, lowcut, highcut, fs, order=4)
    yr = (yr - np.mean(yr)) / np.std(yr)

    outputs = np.array(outputs)
    bpm_out = []
    bpm_out2 = []
    win = 255
    for i in range(2*win, len(outputs), win):
        if (i<len(outputs)) and (i+win<len(outputs)):
            peaks_out, _ = find_peaks(yr[i:i + win], height=0.95)
            _, mmm = hp.process(yr[i:i + win], 30.0)
            bpm_out.append(mmm['bpm'])
            bpm_out2.append(30/(win/len(peaks_out))*win)
    bpm_out, s, m = remove_outliers(bpm_out)
    print("\tGraphing Heart Rate.....")
    outputs = torch.tensor(outputs)
    plt.subplots_adjust(right=0.7)
    x = np.linspace(0, duration, len(bpm_out))
    plt.ylim([0, 140])
    plt.plot(x, bpm_out, alpha=0.7, label='Heart Rate Output', markersize = 4)
    # bmmp_filt and bmp_out are important.
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='large')
    plt.title("Remotely Detected Heart Rate")
    plt.ylabel('HR', fontsize='large')
    plt.xlabel('Time (seconds)', fontsize='large')
    #plt.grid()
    #plt.show()
    plt.savefig(os.path.join('static', 'HRsta.PNG'))
    plt.clf()
    print("Heart Rate Detection Complete!")
    return bpm_out
