import os
import glob
import cv2
import numpy as np
import json
import pandas as pd
from tqdm import tqdm


def face_crop(maindir, path='PURE', save_path='face_crop'):

    if not os.path.exists(save_path+"/"+maindir):
        os.makedirs(save_path+"/"+maindir)
    face_cascade = cv2.CascadeClassifier('/content/drive/MyDrive/Stress Detection/CKs/CK+/haar/haarcascade_frontalface_default.xml')

    sequence_dir = os.path.join(path)

    faces_list = []
    frames = glob.glob(path + '/*.png')
    if len(frames) > 0:
        print(frames)
        image = np.array(cv2.imread(frames[0]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 2)

        (x, y, w, h) = faces[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
        y = y - 30

        prev_y = y
        prew_h = h
        prew_x = x
        prew_w = w

        cropped = image[y:y + int(h / 2), x:x + w]
        faces_list.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

        for fr in tqdm(range(len(frames))):
            image = np.array(cv2.imread(frames[fr]))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 2)

            cropped = image[y:y + int(h * 3 / 2), x:x + w]

            # optional save of bounding rectangle s
            faces_list.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

            cv2.imwrite(save_path+'/'+maindir+'/{0:05d}.png'.format(fr), cropped)

        ff = {'rectangles': faces_list}
        with open(save_path + '/' + maindir + '.json', 'a') as f:
            json.dump(ff, f)

entirecrop = ["06-01","06-03","06-04","06-05","06-06",
              "07-01","07-02","07-04","07-05","07-06",
              "08-01","08-02","08-03","08-04","08-05",
              "09-01","09-02","09-03","09-04","09-05","09-06",
              "10-01","10-02","10-03","10-04","10-05","10-06"]

# for i in entirecrop:
  #face_crop(i, "/content/drive/MyDrive/Stress Detection/PURE/"+i, "/content/drive/MyDrive/Stress Detection/PURECROPPED")