# how to use css in python_ flask
# flask render_template example
 
from flask import Flask, render_template,Response
import cv2
from cv2 import COLOR_BGR2GRAY
import numpy as np
import os
from cvzone.ClassificationModule import Classifier
 
# WSGI Application
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')

mydata = Classifier('data/keras_model.h5', 'data/labels.txt')
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            predict, index = mydata.getPrediction(frame, color=(0,0,255))
            print(predict, index)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
 
if __name__=='__main__':
    app.run(debug = True)