# Adversarial Attack Visualization
Source code for the website and the project to generate adversarial examples to fool common Machine Learning models.

This is the repository for Visualizing and Comparision of Various Adversarial Attacks on user Uploaded Images using a User-Friendly interface using the DNN framework Pytorch, using popular SOTA Pretrained `TorchVision`  ModelZoo. The Following Attacks have been implemented so far and code can be found inside `GAE/attacks.py`

* Fast Gradient Sign Method, Untargeted [1]
* Fast Gradient Sign Method, Targeted [1]
* Basic Iterative Method, Untargeted [2]
* Least Likely Class Iterative Method [2]
* DeepFool, untargeted [3]
* LBFGS, targeted [4]

Coming Soon: Carlini-Wagner l2, and Many More

Installation
------------

Clone the git repository :

```
git clone https://github.com/dsgiitr/adversarial_gen_demo.git
```
Python 3 with Pytorch 1.4.0 is the primary requirement. The `requirements.txt` file contains a listing of required Python packages; to install all requirements, run the following:

```
pip3 install -r requirements.txt
```

Deploying webserver:
--------------------

After downloading the repo, run `python3 app.py`:

```
$ cd adversarial_gen_demo
$ python3 app.py
```

Fire up your browser and navigate to `localhost:5000` or `your_server_url:5000`. Upload any image in JPG format, Select SOTA `torchvision` model and Adversarial Attack strategy. Experiment with the parameters for a particular algorithm and push 'Generate'. After a short while, the server returns Ajax response with Perturbed Image and Perturbation of the Original Image along with Top 5 Classified Labels for the Same. 

Framework
---------
	- Python `Flask`-based server
		- Python backend allows access to Pytorch 
	- Front-end using JQuery and Bootstrap

