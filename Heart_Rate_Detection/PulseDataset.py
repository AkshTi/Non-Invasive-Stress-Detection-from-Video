# Pulse dataset created for training. Contains waveform data and images.
import torch
import glob
import os
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import resample
import os
import glob
import torch.nn.parallel

class PulseDataset(Dataset):

    def __init__(self, sequence_list, root_dir, length, img_height=120, img_width=120, seq_len=1, transform=None):
        seq_list = []
        with open(root_dir + "/" + sequence_list, 'r') as seq_list_file:
            for line in seq_list_file:
                seq_list.append(line.rstrip('\n'))
        self.frames_list = pd.DataFrame()
        for s in seq_list:
            sequence_dir = os.path.join(root_dir, s)
            if sequence_dir[-2:len(sequence_dir)] == '_1':
                fr_list = glob.glob(sequence_dir[0:-2] + '/cropped/*.png')
                fr_list = fr_list[0:len(fr_list) // 2]
            elif sequence_dir[-2:len(sequence_dir)] == '_2':
                fr_list = glob.glob(sequence_dir[0:-2] + '/cropped/*.png')
                fr_list = fr_list[len(fr_list) // 2: len(fr_list)]
            else:
                if os.path.exists(sequence_dir + '/cropped/'):
                    fr_list = glob.glob(sequence_dir + '/*.png')
                else:
                    fr_list = glob.glob(sequence_dir.rstrip() + '/*.png')

            file = open(sequence_dir.rstrip()+".json")
            ref = []
            for line in file:

              if (len(line) > 5 and line[5]=="w"):
                ref.append(line[-3:-1])

            ref = np.array(ref)
            ref_resample = resample(ref, len(fr_list))
            ref_resample = (ref_resample - np.mean(ref_resample)) / (np.max(ref_resample)-np.min(ref_resample))
            self.frames_list = self.frames_list.append(pd.DataFrame({'frames': fr_list, 'labels': ref_resample}))

        self.length = length
        self.seq_len = seq_len
        self.img_height = img_height
        self.img_width = img_width
        self.root_dir = root_dir
        self.transform = transform
        print('Found', self.__len__(), "sequences")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frames = []

        for fr in range(idx, idx+self.seq_len):  # frames from idx to idx+seq_len
            img_name = os.path.join(self.frames_list.iloc[fr, 0])  # path to image
            image = Image.open(img_name)
            image = image.resize((self.img_width, self.img_height))

            if self.transform:
                image = self.transform(image)
            frames.append(image)

        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)
        frames = torch.squeeze(frames, dim=1)
        frames = (frames-torch.mean(frames))/torch.std(frames)*255
        lab = np.array(self.frames_list.iloc[idx:idx + self.seq_len, 1])
        labels = torch.tensor(lab, dtype=torch.float)

        sample = (frames, labels)
        return sample