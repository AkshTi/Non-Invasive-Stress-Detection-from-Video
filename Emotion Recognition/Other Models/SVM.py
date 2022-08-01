from Preprocessing import *
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

accuracy_linear_svm = []
training_accuracy_linear_svm = []
epochs = 50
emotions = ["sadness", "neutral", "anger", "surprise", "disgust", "contempt", "fear", "happy"]
print("Training Linear SVM")
for i in range(epochs):
    print("Creating set %s" %(i + 1)) 
    training_data, training_classes, test_data, test_classes = create_training_test_sets()
    training_data_array = np.array(training_data) 
    test_data_array = np.array(test_data)
    print("Training linear SVM for set %s" %(i+1)) #train SVM
    linear_svm.fit(training_data_array, training_classes)
    prediction_accuracy = linear_svm.score(test_data_array, test_classes)
    print("Accuracy for set %s:" %(i+1), prediction_accuracy*100)
    accuracy_linear_svm.append(prediction_accuracy)
    prediction_labels = linear_svm.predict(test_data)
    result = confusion_matrix(test_classes, prediction_labels)
    df_cm = pd.DataFrame(result, index = [i for i in emotions], columns = [i for i in emotions])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap="Greens")
    img_name = '/content/drive/MyDrive/Stress Detection/CKs/CK+/TrainModels/SVM/svm_confusion_matrix%s.png' %(i + 1)
    plt.savefig(img_name)

xlist = [i for i in range(epochs)]
sn.lineplot(xlist, accuracy_linear_svm)
print("Mean value of accuracies using linear SVM: %s" %(np.mean(accuracy_linear_svm) * 100))
svm_model_name = os.path.join("/content/drive/MyDrive/Stress Detection/CKs/CK+/TrainModels/",  "linear_svm.model")
joblib.dump(linear_svm, svm_model_name, compress = 3)

