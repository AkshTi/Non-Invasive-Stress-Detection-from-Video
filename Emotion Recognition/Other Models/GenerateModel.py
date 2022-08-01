from Preprocessing import *
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sn
import matplotlib.pyplot as plt
from seaborn.palettes import color_palette

final_accuracies = [85, 84.24590163934425, 80.05464480874316 ,56.6120218579235]
plt.title("Models vs. Accuracy")
plt.ylabel("Accuracy (%)")
models = ["CNN", "Linear SVM", "Decision Tree", "AdaBoost"]
plt.xlabel("Models")
sn.barplot(x=models, y=final_accuracies, palette="Greens")
plt.ylim(0,100)