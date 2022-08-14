#Cropping software for PURE dataset (prior to training).
import os
import glob
import cv2
import numpy as np
import json
from tqdm import tqdm


def face_crop(maindir, path='PURE', save_path='face_crop'):

    if not os.path.exists(save_path+"/"+maindir):
        os.makedirs(save_path+"/"+maindir)
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
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
