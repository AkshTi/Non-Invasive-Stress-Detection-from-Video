from LipEyebrowFacialDetection import *
stress_value_list, stress_level_list, fps, total= get_frame("pathToMovie")

#generate the entire plot
import seaborn as sns
import matplotlib.pyplot as plt
xranges = []
i =0
for i in range(0, total//fps): 
  xranges.append(i)
xarray = np.arange(0, total/fps, 1/fps)
#print(xarray)
plt.xlabel("Frames")
plt.ylabel("Stress score")
plt.axhline(y=0.75, color="green", linestyle = "--", linewidth=0.8, label="high stress threshold")
plt.plot(stress_value_list, "mediumaquamarine", linewidth=0.8, label="stress score" )
plt.axhline(y=0.65, color="green", linestyle = "--", linewidth=0.7, label="medium stress threshold")
plt.title("Facial Movement based Stress Detection")
#plt.axhline(y=0.75, color="mediumaquamarine", linestyle = "--", linewidth=0.8)
#plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
plt.legend()
plt.show()