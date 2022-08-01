import glob
from shutil import copyfile, move
import shutil
import os
import stat
import cv2
from os import rename, listdir
from os.path import isfile, join

EMOTIONS = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] 

def copy_tree(src, dst, symlinks = False, ignore = None):
  if not os.path.exists(dst):
    os.makedirs(dst)
    shutil.copystat(src, dst)
  lst = os.listdir(src)
  if ignore:
    excl = ignore(src, lst)
    lst = [x for x in lst if x not in excl]
  for item in lst:
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if symlinks and os.path.islink(s):
      if os.path.lexists(d):
        os.remove(d)
      os.symlink(os.readlink(s), d)
      try:
        st = os.lstat(s)
        mode = stat.S_IMODE(st.st_mode)
        os.lchmod(d, mode)
      except:
        pass # lchmod not available
    elif os.path.isdir(s):
      shutil.copytree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)

# #1. Copy files to two folders
src1 = '/content/drive/MyDrive/Stress Detection/CKs/CK+/extended-cohn-kanade-images/cohn-kanade-images'
dst1 = '/content/drive/MyDrive/Stress Detection/CKs/CK+/source_images'

src2 = '/content/drive/MyDrive/Stress Detection/CKs/CK+/Emotion_labels/Emotion'
dst2 = '/content/drive/MyDrive/Stress Detection/CKs/CK+/source_emotion'

copy_tree(src1, dst1)
copy_tree(src2, dst2)

print('----------1. Finished copying files to two source folders')

# # 2. Create 8 labelled folders
for i in range(len(EMOTIONS)):
    subfolder = '/content/drive/MyDrive/Stress Detection/CKs/CK+/sorted_set/'+ EMOTIONS[i]
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

print('----------2. Finished creating 8 sub folders')

# # 3. Distribute files into 8 folders
participants = glob.glob("/content/drive/MyDrive/Stress Detection/CKs/CK+/source_emotion/*") #Returns a list of all folders with participant numbers
for x in participants:
    part = "%s" %x[-4:] #store current participant number
    print("Part: ", part)
    for sessions in glob.glob("%s/*" %x): #Store list of sessions for current participant
        print("sessions: ", sessions)
        for files in glob.glob("%s/*" %sessions):
            print("files: ", files)
            current_session = files[-29+5:-25+4]
            print("Current session:", current_session)
            file = open(files, 'r')
            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
            sourcefile_emotion = glob.glob("/content/drive/MyDrive/Stress Detection/CKs/CK+/source_images/%s/%s/*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
            sourcefile_neutral = glob.glob("/content/drive/MyDrive/Stress Detection/CKs/CK+/source_images/%s/%s/*" %(part, current_session))[0] #do same for neutral image
            print(sourcefile_neutral[-21:])
            dest_neut = "/sorted_set/neutral/%s" %sourcefile_neutral[-21:] #Generate path to put neutral image
            dest_emot = "/sorted_set/%s/%s" %(EMOTIONS[emotion], sourcefile_emotion[-21:]) #Do same for emotion containing image
            # dest_neut = dest_neut.replace("Stress Detection/CKs/CK+/source_images/", "")
            # dest_emot = dest_emot.replace("Stress Detection/CKs/CK+/source_images/", "")

            print(dest_neut)
            #/sorted_set/neutral/S005/001/S005_001_00000001.png
            copyfile(sourcefile_neutral, "/content/drive/MyDrive/Stress Detection/CKs/CK+" + dest_neut) #Copy file 47
            copyfile(sourcefile_emotion, "/content/drive/MyDrive/Stress Detection/CKs/CK+" +dest_emot) #Copy file

print('----------3. Finished distribute files into 8 sub folders')

# 4. Create 8 sub folders for extracted faces
for i in range(len(EMOTIONS)):
    subfolder = '/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/'+ EMOTIONS[i]
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

print('----------4. Finished creating 8 sub folders for extracted faces')

# # 5. OpenCV to grayscale, crop each image
faceDet = cv2.CascadeClassifier("/content/drive/MyDrive/Stress Detection/CKs/CK+/haar/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("/content/drive/MyDrive/Stress Detection/CKs/CK+/haar/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("/content/drive/MyDrive/Stress Detection/CKs/CK+/haar/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("/content/drive/MyDrive/Stress Detection/CKs/CK+/haar/haarcascade_frontalface_alt_tree.xml")

def detect_faces(emotion):
    files = glob.glob("/content/drive/MyDrive/Stress Detection/CKs/CK+/sorted_set/%s/*" %emotion) #Get list of all images with emotion
    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print("face found in file: %s" %f)
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            try:
                out = cv2.resize(gray, (227, 227)) #Resize face so all images have same size
                cv2.imwrite("/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/%s/%s.jpg" %(emotion, filenumber), out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number
for emotion in EMOTIONS:
    detect_faces(emotion) #Call functiona

print('----------5. Finished OpenCV')

# # 6. Remove extra Neutral faces
index_to_remove = [1,2,5,6,8,10,11,13,14,16,17,18,21,24,25,26,28,30,31,33,34,37,38,41,42,43,44,45,\
                   48,49,52,53,54,56,57,59,60,62,63,65,66,69,71,72,73,75,76,77,78,80,81,83,84,86,87,\
                   90,93,94,95,97,98,99,100,102,103,104,107,108,109,110,111,113,114,117,118,119,122,123,\
                   125,126,127,130,131,132,134,135,136,137,140,141,142,144,145,147,149,150,151,156,157,158,\
                   162,163,165,168,171,172,173,174,177,180,181,184,185,187,189,190,193,194,196,197,198,200,\
                   202,203,205,206,207,209,210,211,213,214,215,217,218,219,221,222,224,225,227,228,229,230,\
                   232,234,236,237,238,240,241,242,244,245,247,249,250,252,253,255,256,258,259,261,262,265,266,\
                   268,270,272,273,274,276,278,279,281,282,284,286,287,289,291,294,295,296,298,299,301,302,304,\
                   306,308]
for i in range(len(index_to_remove)):
    file_name = f'/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/neutral/{index_to_remove[i]}.jpg'
    print(file_name)
    try:
        os.remove(file_name)
    except OSError:
        pass

file_list = os.listdir('/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/neutral')
file_num_list = [int(num.split('.')[0]) for num in file_list]
file_num_list.sort()
print(file_num_list)
 
for j in range(len(file_num_list)):
    current_file = '/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/neutral/'+str(file_num_list[j])+'.jpg'
    new_file = f'/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/neutral/{j}.jpg'

    print(current_file+'  '+new_file)
    rename(current_file, new_file)

print('----------6. Finished Deleting duplicate Neutral faces')


import numpy as np
import random
data = {}

def get_files(emotion):
    files = glob.glob(r"/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/%s/*"%emotion)
    random.shuffle(files)
    training  = files[:int(len(files)*.80)]
    prediction = files[-int(len(files)*.20):]    
    return training, prediction

def makeset():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    
    for emotion in EMOTIONS:
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            training_data.append(gray)
            training_labels.append(EMOTIONS.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(EMOTIONS.index(emotion))
    np.save(r'/content/drive/MyDrive/Stress Detection/CKs/CK+/train_images.npy', [a for a in training_data])
    np.save(r'/content/drive/MyDrive/Stress Detection/CKs/CK+/train_labels.npy', [a for a in training_labels])
    np.save(r'/content/drive/MyDrive/Stress Detection/CKs/CK+/test_images.npy', [a for a in prediction_data])
    np.save(r'/content/drive/MyDrive/Stress Detection/CKs/CK+/test_labels.npy', [a for a in prediction_labels])
makeset()

# 7. Split testing and training sets
SPLIT_RATE = 0.8

subfolders = [name for name in os.listdir('/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset')
            if os.path.isdir(os.path.join('/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset', name))]

for folder_name in subfolders:
    sub_test_folder = os.path.join('/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/testing/'+folder_name)
    if not os.path.exists(sub_test_folder):
        os.makedirs(sub_test_folder) 

    sub_train_folder = os.path.join('/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/training/'+folder_name)
    if not os.path.exists(sub_train_folder):
        os.makedirs(sub_train_folder) 

    all_files = [f for f in listdir('/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/'+folder_name) if isfile(join('dataset/'+folder_name, f))]

    training_files = all_files[:int(len(all_files)*SPLIT_RATE)]

    for item in training_files:
        move('/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/'+folder_name +'/' + item, sub_train_folder+'/'+item)
    
    remain_files = [f for f in listdir('/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/'+folder_name) if isfile(join('dataset/'+folder_name, f))]

    for item in remain_files:
        move('/content/drive/MyDrive/Stress Detection/CKs/CK+/dataset/'+folder_name +'/' + item, sub_test_folder+'/'+item)

print('----------7. Finished Splitting files')