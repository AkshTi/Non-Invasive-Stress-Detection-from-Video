from EntireRec import *
from PulseSampler import *
from EDetect import *
from Common import *

def getEmotion(path, duration):
  print("\nPROCESSING EMOTIONS")
  print("\t resizing videos.....")
  getSizes(path)
  print("\t running model.....")
  emotion_lists = getEmfromVideo(path, duration)
  print("\tEmotion Analysis Complete!\n")
  return emotion_lists

# Emotion Function
# Plots and releases a high/medium/lower stress 

def getEmotionEyebrowDetection(path, duration, fps, frame_count):
  print("\nEYEBROW AND LIP MOVEMENT ANALYSIS")
  print("\t calculating Stress Values")
  stress_value_list, stress_level_list, fps, total= get_frame(path, duration, fps, frame_count)
  time.sleep(4)
  xarray = np.linspace(0, duration, len(stress_value_list))
  plt.xlabel("Time (seconds)")
  plt.ylabel("Score")
  plt.axhline(y=0.75, color="red", linestyle = "--", linewidth=1.1, label="High stress threshold")
  plt.plot(xarray, stress_value_list, linewidth=1.8, label="Score" )
  plt.axhline(y=0.65, color="gold", linestyle = "--", linewidth=1.1, label="Medium stress threshold")
  plt.title("Facial Movement Detection")
  plt.legend()
  plt.savefig(os.path.join('static', 'FacialMovement.png'))
  plt.clf()
  print("Lip and Eyebrow Analysis Complete!\n")
  return stress_value_list

#-----converter

def converter(emotion_array, bpm_list, stress_value_list, duration):
  high_stress = 1.0
  semi_high_stress = 0.8
  medium_stress = 0.5
  low_stress = 0.3
  no_stress = 0.0
  """
  ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] 
  high stress emotions:
  anger, disgust ->high stress
  fear, surprise -> semi_high_stress
  contempt -> medium stress
  sadness -> low_stress
  happy -> no stress
  """

  def getHRSt(bpms):
    if 140 <= bpms:
      return high_stress
    elif bpms >= 120:
      return 0.9
    elif bpms >= 100:
      return semi_high_stress
    elif bpms >= 90:
      return medium_stress
    elif bpms >= 80:
      return low_stress
    else:
      return no_stress

  final_stress_score = []
  count = 0

  emotion_dict = {0:no_stress, 1: high_stress, 2: medium_stress, 3: high_stress, 4 : semi_high_stress, 5: no_stress, 6: low_stress, 7: semi_high_stress}
  heart_rate_dict = {110: high_stress, 100: semi_high_stress, 90: medium_stress, 80: low_stress, 40: no_stress}
  measure_skip = len(emotion_array) // len(bpm_list)

  for i in range(len(emotion_array)):
    if i % measure_skip == 0 and count < len(bpm_list) and i < len(stress_value_list):
      total_val = 0
      total_val += emotion_dict[emotion_array[i]]
      total_val += getHRSt(bpm_list[count])
      total_val += stress_value_list[i]
      final_stress_score.append(total_val)
      count+=1
  xarray = np.linspace(0, duration, len(final_stress_score))
  final_stress_score[0] = 0
  plt.xlabel("Time (seconds)")
  plt.ylabel("Final Stress Score")
  sns.lineplot(x=xarray, y=final_stress_score)
  plt.axhline(y=2.4, color="red", linestyle = "--", linewidth=0.8, label="High Stress")
  plt.axhline(y=1.9, color="gold", linestyle = "--", linewidth=0.7, label="Medium Stress")
  plt.title("Stress of Individual vs. Time")
  plt.ylim([0, 3])
  plt.legend()
  #plt.show()
  plt.savefig(os.path.join('static', 'StressGraph.png'))
  plt.clf()

  return final_stress_score
  
#--------------entire function
def getStressed(videofilename, framedirectory):
    duration, fps, frame_count = getMetrics(videofilename)
    
    # print('AAAAAA', duration, fps, frame_count)
    getPicfromVideo(videofilename, framedirectory)
    emotion_array = getEmotion(framedirectory, duration)
    stress_value_list = stress_value_list = getEmotionEyebrowDetection(videofilename, duration, fps, frame_count)
    bpm_list = getHeartRate(framedirectory, duration)
    print("Combining all 3 outputs into a final stress score.....")
    final_stress_score = converter(emotion_array, bpm_list, stress_value_list, duration)
    print("Contactless Stress Detection Complete! Results are above ^")

    return final_stress_score

#----run

print("Welcome to the Non-Invasive Stress Detector!\n")
print("We measure stress levels using three levels.")
print("[1] Emotions")
print("[2] Heart Rate")
print("[3] Lip and Eyebrow movement\n")
#filepath = input("Enter the directory that your video is located in: ")
#filename = input("\nNow, enter the full name of the video: ")
#print("Video: ", filepath+"\\"+filename)
#img = mpimg.imread(r"C:\Dev\Tools\Python\condabin\stressdetection\Code\IMG_7791.PNG", 0)
#imgplot = plt.imshow(img)
# plt.axis("off")
# plt.show()
#getStressed("C:\Dev\Tools\Python\condabin\stressdetection\Code\Video", "Test.mp4")

#C:\Dev\Tools\Python\condabin\stressdetection\Code
"""set FLASK_ENV=development 
set FLASK_APP=app.py 
flask run
"""
