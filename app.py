import os
import io

from flask import Flask, request, render_template, redirect, jsonify, Response
import base64 as b64

import commons
from attack import fgsm_untargeted
import torch
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__ , template_folder='template')

@app.route('/fgsm_untargeted', methods=['GET'])
def hello_world():
    return render_template('fgsm.html')

@app.route('/hello', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        with open('images/sakshi.png', 'wb') as f:
            data = b64.b64decode(request.form['data'])
            f.write(data)
        return 'Image Saved!'

@app.route('/atk_fgsm_untargeted', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("file not found")
            return redirect(request.url)
        file = request.files['file']
        #epsilon = float(request.args.get('eps'))
        epsilon = float(request.form['eps'])
        model = request.form['model']
        # if not file:
        #     print('no file')
        #     return 0
        img_bytes = file.read()
        top5_original, top5_perturbed, perturbed_image, original_image, perturbation = fgsm_untargeted( model, img_bytes, epsilon, device=torch.device('cpu'))
        # print(original_image.dtype,"shape",original_image.shape)
        Image.fromarray((perturbed_image*255).astype('uint8')).save('perturbed_image.jpg')
        Image.fromarray((original_image*255).astype('uint8')).save('original.jpg')
        # cv2.imwrite('image1.jpg',original_image)
        # cv2.imshow("Image", original_image)
        # cv2.waitKey();
        # per_buffer = cv2.imencode('.jpg', perturbed_image)[1]
        # per_data = b64.b64encode(per_buffer).decode()
        # perturbation_buffer = cv2.imencode('.jpg', perturbation)[1]
        # perturbation_data = b64.b64encode(perturbation_buffer).decode()
        # or_buffer = cv2.imencode('.jpg', original_image)[1]
        # or_data = b64.b64encode(or_buffer).decode()

        or_data = commons.getb64str(original_image).decode()
        perturbation_data = commons.getb64str(perturbation).decode()
        per_data = commons.getb64str(perturbed_image).decode()
        # print(b64.b64encode(or_data))
        # or_data = or_data#.decode('utf8').replace("'",'"')
        # perturbation_data = perturbation_data#.decode('utf8').replace("'",'"')
        # per_data = per_data#.decode('utf8').replace("'",'"')
        # im = Image.fromarray(original_image.astype("uint8"))
        # rawBytes = io.BytesIO()
        # im.save(rawBytes, "PNG")
        # rawBytes.seek(0)
        # with open('images/or_image.png', 'wb') as f:
        #     data = rawBytes.read()
        #     f.write(data)
        print('Success!')
        # img = Image.fromarray(original_image.astype("uint8"))
        # rawBytes = io.BytesIO()
        # img.save(rawBytes, "JPEG")
        # rawBytes.seek(0)
        # img_base64 = base64.b64encode(rawBytes.read())
        # return jsonify({'or_image':str(img_base64)})
        return jsonify({'original':top5_original,'perturbed':top5_perturbed, 'per_image':per_data, 'or_image':or_data, 'perturbation':perturbation_data})

