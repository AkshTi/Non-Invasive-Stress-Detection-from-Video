from Preprocessing import *
import seaborn as sn
import matplotlib.pyplot as plt

#imported from a training session.
final_accuracies = [85, 84.24590163934425, 80.05464480874316 ,56.6120218579235]
plt.title("Models vs. Accuracy")
plt.ylabel("Accuracy (%)")
models = ["CNN", "Linear SVM", "Decision Tree", "AdaBoost"]
plt.xlabel("Models")
sn.barplot(x=models, y=final_accuracies, palette="Greens")
plt.ylim(0,100)