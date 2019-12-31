import os
import io

from flask import Flask, request, render_template, redirect, jsonify
import base64 as b64

import commons
from attack import fgsm_untargeted
import torch

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
        top5_original, top5_perturbed, perturbed_image = fgsm_untargeted(img_bytes, epsilon, device=torch.device('cpu'))
        #class_id, class_name = get_prediction(image_bytes=img_bytes)
        #return render_template('result.html', class_id=class_id,
        #                       class_name=class_name)
        #response = app.response_class(response)
        return jsonify({'original':top5_original,'perturbed':top5_perturbed,'perturbed_image':perturbed_image})#, jsonify(top5_perturbed)#, perturbed_image #render_template('index.html')

