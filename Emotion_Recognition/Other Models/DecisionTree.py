emotions = ["sadness", "neutral", "anger", "surprise", "disgust", "contempt", "fear", "happy"]
from Preprocessing import *
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import joblib
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


accuracy_decision_tree = []
training_accuracy_decision_tree = []
print("Training Decision Tree")
for i in range(0, 10):
    #Creating sets by randomly sampling 80:20
    print("Creating set %s" %(i + 1)) 
    training_data, training_classes, test_data, test_classes = create_training_test_sets()
    #Obtaining the numpy array for the training and test dataset for the classifier
    training_data_array = np.array(training_data) 
    test_data_array = np.array(test_data)
    print("Training Decision Tree for set %s" %(i + 1))
    #Train Decision Tree
    decision_tree.fit(training_data_array, training_classes)
    prediction_accuracy = decision_tree.score(test_data_array, test_classes)
    training_prediction_accuracy = decision_tree.score(training_data_array, training_classes)
    training_accuracy_decision_tree.append(training_prediction_accuracy)
    print("Testing Accuracy for set %s:" %(i + 1), prediction_accuracy*100)
    print("Training Accuracy for set %s:" %(i+1), training_prediction_accuracy*100)
    #Store accuracy in a list
    accuracy_decision_tree.append(prediction_accuracy)
    prediction_labels = decision_tree.predict(test_data)
    #print the confusion matrix
    result = confusion_matrix(test_classes, prediction_labels)
    df_cm = pd.DataFrame(result, index = [i for i in emotions], columns = [i for i in emotions])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap = "Greens")
    img_name = '/content/drive/MyDrive/Stress Detection/CKs/CK+/TrainModels/Decision Tree/dt_confusion_matrix%s.png' %(i + 1)
    plt.savefig(img_name)
#Obtain mean accuracy of the 10 iterations
print("Mean value of accuracies using Decision Tree: %s" %(np.mean(accuracy_decision_tree) * 100))
dt_model_name = os.path.join("content/drive/MyDrive/Stress Detection/CKs/CK+/TrainModels/",  "decision_tree.model")
joblib.dump(decision_tree, dt_model_name, compress = 3)
