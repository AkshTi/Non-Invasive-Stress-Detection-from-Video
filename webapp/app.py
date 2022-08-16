from Integration import *
from flask import request
from flask import Flask, render_template, jsonify, request
import os, json
from werkzeug.utils import secure_filename
import tempfile, datetime
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

OUTDIR = app.instance_path
os.makedirs(OUTDIR, exist_ok=True)


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
    print('** NEW VIDEO UPLOADED **')
    if request.method == 'POST':

        ### CREATE AND GRAB FOLDER STRUCTURE
        USERDIR, FRAMESDIR, PLOTSDIR = createOutputFolders(OUTDIR)

        f = request.files['file']
        videofilename = 'uploaded.webm'
        videopath = os.path.join(USERDIR, videofilename)
        f.save(videopath)

        if videopath.endswith('webm'): # ALWAYS THE CASE SINCE ONLY WEBM is SUPPORTED IN JS
          convertedvideopath = videopath.replace('.webm', '.mp4')
          os.system('ffmpeg -i ' + videopath + ' -filter:v fps=30 ' + convertedvideopath)
          videopath = convertedvideopath


        final_stress_score, heart_rates, emotions, facial_movements = getStressed( videopath, 
                                                                                   FRAMESDIR,
                                                                                   PLOTSDIR)

        data = {
            "StressText" : "Your Stress is .....",
            "StressScore": final_stress_score,
            "HeartRates": [float(v) for v in heart_rates],
            "Emotions": [float(v) for v in emotions],
            "FacialMovements": [float(v) for v in facial_movements]
        }  
        jsonoutput = json.dumps(data)


        jsonfile = os.path.join(USERDIR, 'results.json')
        with open(jsonfile, 'w') as f:
          f.write(jsonoutput)

        return jsonify(data) # we need the flask response here

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