import os
import io

from flask import Flask, request, render_template, redirect, jsonify, Response
import base64 as b64

import commons
import attack
import torch
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__ , template_folder='template')

@app.route('/fgsm_untargeted', methods=['GET'])
def fgsm():
    return render_template('fgsm.html')

@app.route('/fgsm_targeted', methods=['GET'])
def fgsm_targeted():
    return render_template('fgsm_targeted.html')

@app.route('/basic_iterative', methods=['GET'])
def basic_iterative():
    return render_template('basic_iterative.html')

@app.route('/iterative_ll_class', methods=['GET'])
def iterative_ll_class():
    return render_template('iterative_ll_class.html')

@app.route('/deep_fool', methods=['GET'])
def deep_fool():
    return render_template('deep_fool.html')

@app.route('/lbfgs', methods=['GET'])
def lbfgs():
    return render_template('lbfgs.html')

@app.route('/atk_fgsm_untargeted', methods=['POST'])
def atk_fgsm_untargeted():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("file not found")
            return redirect(request.url)
        file = request.files['file']
        epsilon = float(request.form['eps'])
        model = request.form['model']
        if not file:
            print('no file')
            return 0
        img_bytes = file.read()
        top5_original, top5_perturbed, perturbed_image, original_image, perturbation = attack.fgsm_untargeted( model, img_bytes, epsilon, device=torch.device('cpu'))
        Image.fromarray((perturbed_image*255).astype('uint8')).save('perturbed_image.jpg')
        Image.fromarray((original_image*255).astype('uint8')).save('original.jpg')
        # cv2.imwrite('image1.jpg',original_image)
        # cv2.imshow("Image", original_image)
        # cv2.waitKey();
        or_data = commons.getb64str(original_image).decode()
        perturbation_data = commons.getb64str(perturbation).decode()
        per_data = commons.getb64str(perturbed_image).decode()
        print('Success!')
        return jsonify({'original':top5_original,'perturbed':top5_perturbed, 'per_image':per_data, 'or_image':or_data, 'perturbation':perturbation_data})

@app.route('/atk_fgsm_targeted', methods=['POST'])
def atk_fgsm_targeted():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("file not found")
            return redirect(request.url)
        file = request.files['file']
        target = int(request.form['target'])
        epsilon = float(request.form['eps'])
        model = request.form['model']
        if not file:
            print('no file')
            return 0
        img_bytes = file.read()
        top5_original, top5_perturbed, perturbed_image, original_image, perturbation = attack.fgsm_targeted( model, img_bytes, epsilon, target, device=torch.device('cpu'))
        or_data = commons.getb64str(original_image).decode()
        perturbation_data = commons.getb64str(perturbation).decode()
        per_data = commons.getb64str(perturbed_image).decode()
        print('Success!')
        return jsonify({'original':top5_original,'perturbed':top5_perturbed, 'per_image':per_data, 'or_image':or_data, 'perturbation':perturbation_data})

@app.route('/atk_basic_iterative', methods=['POST'])
def atk_basic_iterative():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("file not found")
            return redirect(request.url)
        file = request.files['file']
        epsilon = float(request.form['eps'])
        alpha = float(request.form['alpha'])
        num_iter = int(request.form['num_iter'])
        model = request.form['model']
        if not file:
            print('no file')
            return 0
        img_bytes = file.read()
        top5_original, top5_perturbed, perturbed_image, original_image, perturbation = attack.basic_iterative( model, img_bytes, alpha, epsilon, num_iter, device=torch.device('cpu'))
        or_data = commons.getb64str(original_image).decode()
        perturbation_data = commons.getb64str(perturbation).decode()
        per_data = commons.getb64str(perturbed_image).decode()
        print('Success!')
        return jsonify({'original':top5_original,'perturbed':top5_perturbed, 'per_image':per_data, 'or_image':or_data, 'perturbation':perturbation_data})

@app.route('/atk_iterative_ll_class', methods=['POST'])
def atk_iterative_ll_class():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("file not found")
            return redirect(request.url)
        file = request.files['file']
        epsilon = float(request.form['eps'])
        alpha = float(request.form['alpha'])
        num_iter = int(request.form['num_iter'])
        model = request.form['model']
        if not file:
            print('no file')
            return 0
        img_bytes = file.read()
        top5_original, top5_perturbed, perturbed_image, original_image, perturbation = attack.iterative_ll_class( model, img_bytes, alpha, epsilon, num_iter, device=torch.device('cpu'))
        or_data = commons.getb64str(original_image).decode()
        perturbation_data = commons.getb64str(perturbation).decode()
        per_data = commons.getb64str(perturbed_image).decode()
        print('Success!')
        return jsonify({'original':top5_original,'perturbed':top5_perturbed, 'per_image':per_data, 'or_image':or_data, 'perturbation':perturbation_data})

@app.route('/atk_deep_fool', methods=['POST'])
def atk_deep_fool():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("file not found")
            return redirect(request.url)
        file = request.files['file']
        max_iter = int(request.form['max_iter'])
        model = request.form['model']
        if not file:
            print('no file')
            return 0
        img_bytes = file.read()
        top5_original, top5_perturbed, perturbed_image, original_image, perturbation = attack.deep_fool( model, img_bytes, max_iter, device=torch.device('cuda'))
        or_data = commons.getb64str(original_image).decode()
        perturbation_data = commons.getb64str(perturbation).decode()
        per_data = commons.getb64str(perturbed_image).decode()
        print('Success!')
        return jsonify({'original':top5_original,'perturbed':top5_perturbed, 'per_image':per_data, 'or_image':or_data, 'perturbation':perturbation_data})

@app.route('/atk_lbfgs', methods=['POST'])
def atk_lbfgs():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("file not found")
            return redirect(request.url)
        file = request.files['file']
        max_iter = int(request.form['max_iter'])
        target = int(request.form['target'])
        bin_search_steps = int(request.form['bin_search_steps'])
        c = float(request.form['c'])
        model = request.form['model']
        const_upper = float(request.form['const_upper'])
        if not file:
            print('no file')
            return 0
        img_bytes = file.read()
        top5_original, top5_perturbed, perturbed_image, original_image, perturbation = attack.lbfgs( model, img_bytes, target, c, bin_search_steps, max_iter, const_upper, device=torch.device('cuda'))
        or_data = commons.getb64str(original_image).decode()
        perturbation_data = commons.getb64str(perturbation).decode()
        per_data = commons.getb64str(perturbed_image).decode()
        print('Success!')
        return jsonify({'original':top5_original,'perturbed':top5_perturbed, 'per_image':per_data, 'or_image':or_data, 'perturbation':perturbation_data})