# Adversarial Lab
Source code for the website and the project to generate adversarial examples to fool common Machine Learning models.

This is the repository for Visualizing and Comparision of Various Adversarial Attacks on user Uploaded Images using a User-Friendly interface using the DNN framework Pytorch, using popular SOTA Pretrained `TorchVision`  ModelZoo. The Following Attacks have been implemented so far and code can be found inside `GAE/attacks.py`

1. FGSM
	* Fast Gradient Sign Method, Untargeted
	* Fast Gradient Sign Method, Targeted
2. Iterative
	* Basic Iterative Method, Untargeted
	* Least Likely Class Iterative Method
3. DeepFool, untargeted
4. LBFGS, targeted

Coming Soon: Carlini-Wagner l2, and Many More

<center><img src="https://dsgiitr.com/images/work/adversarial_example.gif"></center>

Installation
------------

Clone the git repository :

```git
git clone https://github.com/dsgiitr/adversarial_lab.git
```
Python 3 with Pytorch 1.4.0 is the primary requirement. The `requirements.txt` file contains a listing of required Python packages; to install all requirements, run the following:

```bash
pip3 install -r requirements.txt
```

Deploying webserver:
--------------------

After downloading the repo, run `flask run`:

```bash
$ cd adversarial_lab
$ flask run
```

Fire up your browser and navigate to `localhost:5000` or `your_server_url:5000`. Upload any image in JPG format, Select SOTA `torchvision` model and Adversarial Attack strategy. Experiment with the parameters for a particular algorithm and push 'Generate'. After a short while, the server returns Ajax response with Perturbed Image and Perturbation of the Original Image along with Top 5 Classified Labels for the Same. 

GAE
---

It is a Pytorch Library containing Simple and Fast implementations of Adversarial Attack Strategies using Pytorch. Cleverhans-Future and Advertorch can be referred for proper Robust implementations. GAE is easy to understand and only Processes a single image file at a time (as of now). Usage of the Following can be found on `Attacks Tutorial on Imagenet.ipynb` notebook.


Framework
---------
	- Python `Flask`-based server
		- Python backend allows access to Pytorch 
	- Front-end using JQuery and Bootstrap
