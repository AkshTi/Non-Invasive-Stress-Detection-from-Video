emotions = ["sadness", "neutral", "anger", "surprise", "disgust", "contempt", "fear", "happy"]
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import sys

from Preprocessing import *
accuracy_ada_boost = []
print("Training AdaBoost Classifier")
for i in range(0, 10):
    print("Creating set %s" %(i + 1)) 
    training_data, training_classes, test_data, test_classes = create_training_test_sets()
    training_data_array = np.array(training_data) 
    test_data_array = np.array(test_data)
    print("Training AdaBoost for set %s" %(i + 1))
    ada_boost.fit(training_data_array, training_classes)
    prediction_accuracy = ada_boost.score(test_data_array, test_classes)
    print("Accuracy for set %s:" %(i + 1), prediction_accuracy*100)
    accuracy_ada_boost.append(prediction_accuracy)
    prediction_labels = ada_boost.predict(test_data)
    #print the confusion matrix
    result = confusion_matrix(test_classes, prediction_labels)
    df_cm = pd.DataFrame(result, index = [i for i in emotions], columns = [i for i in emotions])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap = "Greens")
    img_name = '/content/drive/MyDrive/Stress Detection/CKs/CK+/TrainModels/AdaBoost/adb_confusion_matrix%s.png' %(i + 1)
    plt.savefig(img_name)
#Obtain mean accuracy of the 10 iterations
print("Mean value of accuracies using AdaBoost: %s" %(np.mean(accuracy_ada_boost) * 100))
adb_model_name = os.path.join("content/drive/MyDrive/Stress Detection/CKs/CK+/TrainModels/",  "ada_boost.model")
joblib.dump(ada_boost, adb_model_name, compress = 3)