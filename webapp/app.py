from Integration import *
from flask import request
from flask import Flask, render_template, jsonify, request
import os
from werkzeug.utils import secure_filename
import tempfile
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path,'videofiles'),exist_ok=True)


#app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

@app.route('/')
def home():
    return render_template("cam.html")

@app.route('/returnjson', methods = ['GET'])
def ReturnJSON():
    if(request.method == 'GET'):
        data = {
            "StressText" : "Your Stress is .....",
            "ImageURL" :  os.path.join('static', 'StressGraph.png')
        }  
        return jsonify(data)

@app.route('/stress', methods=['POST'])
def detect_stress():
    print('GGGGGG')
    if request.method == 'POST':

        f = request.files['file']

        handle, videofilename = tempfile.mkstemp(suffix='.webm', dir=os.path.join(app.instance_path, 'videofiles'))
        f.save(videofilename)



        framedirectory = tempfile.mkdtemp(prefix=videofilename, dir=os.path.join(app.instance_path, 'frames'))
 

        if videofilename.endswith('webm'):
          os.system('ffmpeg -i ' + videofilename + ' ' + videofilename.replace('.webm', '.mp4'))

        videofilename = videofilename.replace('.webm', '.mp4')

        final_stress_score, heart_rates, emotions, facial_movements = getStressed( videofilename, 
                                          framedirectory)

        #image1=os.path.join('static', 'Emot.PNG')
        #image2=os.path.join('static', 'FacialMovement.PNG')
        #image3=os.path.join('static', 'HRsta.PNG')
        #image4=os.path.join('static', 'StressGraph.PNG')
        #return render_template("result.html", image1 = image1, image2 = image2, image3 = image3, image4 = image4 )
        data = {
            "StressText" : "Your Stress is .....",
            "StressScore": final_stress_score,
            "HeartRates": [float(v) for v in heart_rates],
            "Emotions": [float(v) for v in emotions],
            "FacialMovements": [float(v) for v in facial_movements],
            "ImageURL" :  os.path.join('static', 'StressGraph.png')
        }  
        return jsonify(data)

@app.route('/stresstest', methods=['POST'])
def stresstest():
    if request.method == 'POST':
        selected_option = request.form['videos']
        getStressed(os.path.join('.','static','video'), selected_option, os.path.join("instance"))
        data = {
            "StressText" : "Your Stress is .....",
            "ImageURL" :  os.path.join('static', 'StressGraph.png')
        }  
        return jsonify(data)        