#!/usr/bin/env python3

from Integration import *

import os, sys, shutil
from werkzeug.utils import secure_filename
import tempfile, time
import matplotlib
matplotlib.use('Agg')


VIDEO = sys.argv[1]
print("Loading video", VIDEO)
 
OUTDIR = 'instance'
os.makedirs(OUTDIR, exist_ok=True)


### CREATE AND GRAB FOLDER STRUCTURE
USERDIR, FRAMESDIR, PLOTSDIR = createOutputFolders(OUTDIR)

videofilename = 'uploaded.webm'
videopath = os.path.join(USERDIR, videofilename)
shutil.copy(VIDEO, videopath)

if videopath.endswith('webm'): # ALWAYS THE CASE SINCE ONLY WEBM is SUPPORTED IN JS
  convertedvideopath = videopath.replace('.webm', '.mp4')
  os.system('ffmpeg -i ' + videopath + ' -filter:v fps=30 -b:v 3M  -minrate 3M -maxrate 3M  -bufsize 3M ' + convertedvideopath)
  videopath = convertedvideopath

t0 = time.time()
final_stress_score, heart_rates, emotions, facial_movements = getStressed( videopath, 
                                                                           FRAMESDIR,
                                                                           PLOTSDIR)

print('***PREDICTION TIME***', time.time()-t0, 'seconds..')


data = {
    "StressText" : "Your Stress is .....",
    "StressScore": final_stress_score,
    "HeartRates": [float(v) for v in heart_rates],
    "Emotions": [float(v) for v in emotions],
    "FacialMovements": [float(v) for v in facial_movements]
}  
jsonoutput = json.dumps(data)


jsonfile = os.path.join(USERDIR, 'results.json')
with open(jsonfile, 'w') as f:
  f.write(jsonoutput)

print(' **** ALL DONE! SAYONARA!!! **** ')
