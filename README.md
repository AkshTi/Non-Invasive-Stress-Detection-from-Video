# Non-Invasive-Stress-Detection-from-Video
We present a non-invasive stress detector that identifies an individual's stress from video. It takes a multimodal approach to evaluate stress features through gacial recordings, and alerts the user in case they experience medium to high stress.

---
# How To Use

Download `Common.py`, `EDetect.py`, `EntireRec.py`, `Integration.py`, `PulseSampler.py`, and `app.py` to the same directory. 
Download all model files to the same directory, as well as the the test inputs. 
When running, chnage each path to the directory of the test input, and run **getStressed** in `Integration.py`.

---

# Testing and Training
**Emotion Recognition** includes the several models we trained and tested. Simply run the testing and training files for the respective model you choose.

The Emotion Recognition model is trained off the Extended Cohn Kanade Dataset. 

**Heart Rate Detection** 

The heart rate detection testing and trainnig process relies on the PURE heart rate dataset.

After downloading the dataset, simply replace the directories in these files with that to your PURE dataset.

`utilities.py` -> A series of helper functions.

`PURECrop.py` -> Crops each image in the PURE Dataset.

`PhysNet.py` -> Spatiotemporal model

`PulseDataset.py` -> Framework to hold data.

Download the above and run `TrainHR` for training.
For testing, use `TestHR`.

**Facial Feature Analysis**

The Facial feature detection represents a heuristic approach, and relies on two main files.

`LipEyebrowFacialDetection.py` is the main file.

`LipTest.py` serves mainly for visualization.
