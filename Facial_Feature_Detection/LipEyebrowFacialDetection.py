# Code to run the facial feature analysis.
import numpy as np
import dlib
import cv2
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils

global points, points_lip, emotion_classifier

# determines distance between landmark points
def ebdist(leye,reye):
    eyedist = dist.euclidean(leye,reye)
    points.append(int(eyedist))
    return eyedist

# determines mouth size (distance between upper and lower lip)
def lpdist(l_lower,l_upper):
    lipdist = dist.euclidean(l_lower, l_upper)
    points_lip.append(int(lipdist))
    return lipdist
    
# values are put in formula and output with a movement-based stress score.
def normalize_values(points,disp,points_lip,dis_lip):
    normalize_value_lip = abs(dis_lip - np.min(points_lip))/abs(np.max(points_lip) - np.min(points_lip))
    normalized_value_eye =abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    normalized_value =( normalized_value_eye + normalize_value_lip)/2
    stress_value = (np.exp(-(normalized_value)))
    if stress_value>=0.75:
        stress_label="High Stress"
    else:
        stress_label="Low Stress"
    return stress_value,stress_label
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/content/drive/MyDrive/Stress Detection/CKs/CK+/shape_predictor_68_face_landmarks.dat")
points = []
points_lip = []
def get_frame(directory): 
    stress_value_list = []
    stress_level_list = []
    cap = cv2.VideoCapture(directory)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        if frame is not None:
          frame = imutils.resize(frame, width=500,height=500)
        
          (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
          (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
          (l_lower, l_upper) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

          gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
          count +=1
          detections = detector(gray,0) #initialize the detector
          for detection in detections:
            shape = predictor(frame,detection)
            shape = face_utils.shape_to_np(shape)
                
            leyebrow = shape[lBegin:lEnd]
            reyebrow = shape[rBegin:rEnd]
            openmouth = shape[l_lower:l_upper]
                
            reyebrowhull = cv2.convexHull(reyebrow)
            leyebrowhull = cv2.convexHull(leyebrow)
            openmouthhull = cv2.convexHull(openmouth) # figuring out convex shape when lips opened
            lipdist = lpdist(openmouthhull[-1],openmouthhull[0])
            eyedist = ebdist(leyebrow[-1], reyebrow[0])

            stress_value,stress_label = normalize_values(points,eyedist, points_lip, lipdist)
            print(str(count) + "/" + str(total))
            stress_value_list.append(stress_value)
            stress_level_list.append(stress_label)
            
            if count==total:
              cap.release()
    return stress_value_list, stress_level_list, fps, total
 