
import torch
import torch.nn as nn

import glob
import os
import torch
import numpy as np
import math
import torch.nn.functional as F
import os
import glob
import time
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import heartpy as hp
import PhysNet
import PulseDataset
from utilities import *
from PhysNet import *
resume = "/content/drive/MyDrive/Stress Detection/PURECROPPED/s_Drop_3d_32_14.tar" #'/content/drive/MyDrive/Stress Detection/PURE/Drop_3d_128_14.tar'
print("initialize model...")

seq_len = 32 
model = PhysNet(seq_len)
model = torch.nn.DataParallel(model)
model.cuda()
ss = sum(p.numel() for p in model.parameters())
print('num params: ', ss)
if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(resume))

sequence_list = "test_seq.txt"
root_dir = '/content/drive/MyDrive/Stress Detection/PURECROPPED/'
seq_list = []
end_indexes_test = []
with open(root_dir + "/" + sequence_list, 'r') as seq_list_file:
    for line in seq_list_file:
        seq_list.append(line.rstrip('\n'))

# seq_list = ['test_static']
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
            fr_list = glob.glob(sequence_dir + '/cropped/*.png')
        else:
            fr_list = glob.glob(sequence_dir + '/*.png')
    print(fr_list)
    end_indexes_test.append(len(fr_list))

end_indexes_test = [0, *end_indexes_test]
# print(end_indexes_test)

sampler_test = PulseSampler(end_indexes_test, seq_len, False)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

pulse_test = PulseDataset(sequence_list, root_dir, seq_len=seq_len,
                                        length=len(sampler_test), transform=transforms.Compose([
                                                                                            transforms.ToTensor(),
                                                                                            normalize]))
val_loader = torch.utils.data.DataLoader(pulse_test, batch_size=1, shuffle=False, sampler=sampler_test, pin_memory=True)

model.eval()
criterion = NegPearson()
criterion = criterion.cuda()

outputs = []
reference_ = []
loss_avg = []
loss_avg2 = []
count = 0
start = time.time()
for k, (net_input, target) in enumerate(val_loader):
    net_input = net_input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    with torch.no_grad():
        output, x_visual, x, t = model(net_input)
        # print("output.shape ", output.shape)
        # print("net_input.shape ", net_input.shape)
        # print("target.shape ", target.shape)
        outputs.append(output[0])
        reference_.append(target[0])
        count+=1
        print("Finished: " + str(count) + "/")

end = time.time()
print(end-start, len(val_loader))
outputs = torch.cat(outputs)

outputs = (outputs - torch.mean(outputs)) / torch.std(outputs)
outputs = outputs.tolist()

reference_ = torch.cat(reference_)
reference_ = (reference_-torch.mean(reference_))/torch.std(reference_)
reference_ = reference_.tolist()
print (np.mean(reference_))
print (np.mean(outputs))
fs = 30
lowcut = 1
highcut = 3

yr = butter_bandpass_filter(outputs, lowcut, highcut, fs, order=4)
yr = (yr - np.mean(yr)) / np.std(yr)

plt.subplots_adjust(right=0.7)
#plt.plot(outputs, alpha=0.7, label='Network\n output')
plt.plot(yr, label='Network\n output')
plt.plot(reference_, '--', label='reference\n PPG')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='large')
plt.ylabel('Amplitude', fontsize='large', fontweight='semibold')
plt.xlabel('Time', fontsize='large', fontweight='semibold')
plt.grid()
plt.xlim([350, 550])
plt.ylim([-2, 3])
print("out/ref")
#print(outputs.shape)
print(len(reference_))
plt.savefig('3d.svg', bbox_inches='tight')
plt.show()
#reference_ = [x for x in reference_ if math.isnan(x)==False]
outputs = np.array(outputs)


bpm_ref = []
bpm_out = []
bmmp_filt = []
bpm_out2 = []
hrv_ref = []
hrv_out = []
print(len(reference_))
win = 255
for i in range(2*win, len(reference_), win):
    print("i :" + str(i) + "i + win: " + str(i+win))
    if (i<len(reference_)) and (i+win<len(reference_)):
      peaks, _ = find_peaks(reference_[i:i+win], distance=20, height=0.9)
      peaks_out, _ = find_peaks(yr[i:i + win], height=0.95)
      _, measures2 = hp.process(reference_[i:i+win], 30.0)
      _, mmm = hp.process(yr[i:i + win], 30.0)
      if len(peaks) !=0 and len(peaks_out) !=0:
        if math.isnan(30/(win/len(peaks))*win)==False and math.isnan(measures2['bpm'])== False and math.isnan(30/(win/len(peaks_out))*win)==False and math.isnan(mmm['bpm'])==False:
          bpm_ref.append(30/(win/len(peaks))*win)
          bmmp_filt.append(measures2['bpm'])
          bpm_out.append(mmm['bpm'])
          bpm_out2.append(30/(win/len(peaks_out))*win)
print(len(bpm_ref))
# print(bpm_filt)
# print(peaks_out)
# corr, _ = pearsonr(bmmp_filt, bpm_out)
# c = np.corrcoef(bmmp_filt, bpm_out)
# cc = np.corrcoef(bpm_ref, bpm_out2)
# ccc = np.corrcoef(bmmp_filt, bpm_out2)
# print("ref/out")
# print(bpm_ref)
# print(bmmp_filt)
# print('Correlation:', c)
print (type(reference_))
print (type(output))
#print (reference_.shape)
print (output.shape)

plt.subplots_adjust(right=0.7)
print("max reference", max(reference_))
print("min reference_", min(reference_))
print("max output: ", max(output))
print("min output: ", min(output))
plt.scatter(reference_, outputs)
plt.xlabel("Reference", fontsize = "large", fontweight = "semibold")
plt.ylabel("Output", fontsize = "large", fontweight = "semibold")
plt.show()

plt.subplots_adjust(right=0.7)
time = np.arange(0, 3, 1 / fs)
fourier_transform = np.fft.rfft(outputs)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, fs / 2, len(power_spectrum))
plt.semilogy(frequency, power_spectrum, label='Network output')

fourier_transform = np.fft.rfft(reference_)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
plt.xlim(-0.1, 10)
plt.ylim(10e-6, 10e6)
plt.semilogy(frequency, power_spectrum, label='reference\n PPG')
plt.ylabel('|A(f)|', fontsize='large', fontweight='semibold')
plt.xlabel('Frequency f [Hz]', fontsize='large', fontweight='semibold')
plt.title('Power frequency spectrum')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

reference_ = torch.tensor(reference_)
outputs = torch.tensor(outputs)

criterionMSE = nn.MSELoss()
criterionMAE = nn.L1Loss()
mse = criterionMSE(reference_, outputs)
rmse = torch.sqrt(mse)
mae = criterionMAE(reference_, outputs)
se = torch.std(outputs-reference_)/np.sqrt(outputs.shape[0])
print("MAE: ", mae, "MSE: ", mse, "RMSE: ", rmse, "SE:", se)