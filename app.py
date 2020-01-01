import os
import io

from flask import Flask, request, render_template, redirect, jsonify, Response
import base64 as b64

import commons
from attack import fgsm_untargeted
import torch
import cv2

app = Flask(__name__ , template_folder='template')

@app.route('/b')
def hello_world():
    return render_template('index.html')

@app.route('/hello', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        with open('images/sakshi.png', 'wb') as f:
            data = b64.b64decode(request.form['data'])
            f.write(data)
        return 'Image Saved!'

@app.route('/fgsm_untargeted', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        epsilon = float(request.args.get('eps'))
        if not file:
            return 0
        img_bytes = file.read()
        top5_original, top5_perturbed, perturbed_image, original_image = fgsm_untargeted(img_bytes, epsilon, device=torch.device('cpu'))
        per_data = cv2.imencode('.jpg', perturbed_image)[1]
        per_data = b64.b64encode(per_data).decode("utf-8")
        or_data = cv2.imencode('.jpg', perturbed_image)[1]
        or_data = b64.b64encode(or_data).decode("utf-8")
        return jsonify({'original':top5_original,'perturbed':top5_perturbed, 'per_image':per_data, 'or_image':or_data})

