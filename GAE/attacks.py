import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = ['fgsm', 'fgsm_targeted', 'basic_iterative', 'iterative_ll_class', 'deep_fool']

def fgsm( model, image, epsilon, device=torch.device('cpu')):
	"""Generates Adv Examples for the Batch"""
	image = image.to(device)
	model = model.to(device)
	image.requires_grad = True
	original_output = model(image)
	_, target = torch.max(original_output, 1)
	loss = F.nll_loss(original_output, target)
	model.zero_grad()
	loss.backward()

	data_grad = image.grad.data

	perturbed_image = perturbe(image, epsilon, data_grad)

	perturbed_output = model(perturbed_image)

	image.requires_grad = False
	original_output, perturbed_output = original_output.detach(), perturbed_output.detach()
	return original_output, perturbed_output, perturbed_image.detach()

# FGSM attack code
def perturbe(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    #perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def fgsm_targeted( model, image, epsilon, target, device=torch.device('cpu')):
	"""Generates Adv Examples for the Batch"""
	image, target = image.to(device), target.to(device)
	model = model.to(device)
	image.requires_grad = True
	original_output = model(image)

	loss = F.nll_loss(original_output, target)
	loss = -loss
	model.zero_grad()
	loss.backward()

	data_grad = image.grad.data

	perturbed_image = perturbe(image, epsilon, data_grad)

	perturbed_output = model(perturbed_image)

	image.requires_grad = False
	original_output, perturbed_output = original_output.detach(), perturbed_output.detach()
	return original_output, perturbed_output, perturbed_image.detach()

def basic_iterative( model, image, alpha, epsilon, num_iter=None, device=torch.device('cpu')):
	"""https://arxiv.org/pdf/1607.02533.pdf"""
	#if min value not provided
	if num_iter is None:
		num_iter = int(min(epsilon+4, 1.25*epsilon))

	image = image.to(device)
	model = model.to(device)
	image.requires_grad = True
	original_output = model(image)
	_, target = torch.max(original_output, 1)
	X_adv = image
	for i in range(num_iter):
		output = model(X_adv)
		loss = F.nll_loss(output, target)
		model.zero_grad()
		loss.backward()
		data_grad = image.grad.data
		X_adv = clip(image, perturbe(image, alpha, data_grad), epsilon)
	perturbed_output = model(X_adv)

	image.requires_grad = False
	original_output, perturbed_output = original_output.detach(), perturbed_output.detach()
	return original_output, perturbed_output, X_adv.detach()

def iterative_ll_class( model, image, alpha, epsilon, num_iter=None, device=torch.device('cpu')):
	"""https://arxiv.org/pdf/1607.02533.pdf"""
	#if min value not provided
	if num_iter is None:
		num_iter = int(min(epsilon+4, 1.25*epsilon))

	image = image.to(device)
	model = model.to(device)
	image.requires_grad = True
	original_output = model(image)
	_, target = torch.min(original_output, 1)
	X_adv = image
	for i in range(num_iter):
		output = model(X_adv)
		loss = F.nll_loss(output, target)
		loss = -loss
		model.zero_grad()
		loss.backward()
		data_grad = image.grad.data
		X_adv = clip(image, perturbe(image, alpha, data_grad), epsilon)
	perturbed_output = model(X_adv)

	image.requires_grad = False
	original_output, perturbed_output = original_output.detach(), perturbed_output.detach()
	return original_output, perturbed_output, X_adv.detach()

def deep_fool( model, image, max_iter=10, device=torch.device('cpu')):
    """Generates Adv Examples for the Batch"""
    image = image.to(device)
    model = model.to(device)
    #image.requires_grad = True
    xi = image.clone().detach().requires_grad_()
    model.zero_grad()
    output = model(xi)
    original_output = output.clone().detach()
    _, indices = torch.sort(output[0], descending=True)
    index_list = [i.item() for i in indices.flatten()]
    index_list = index_list[:10]
    target = index_list[0]
    #_, target = torch.max(output, 1)
    ri = torch.empty_like(image, requires_grad=False)
    kxi = kx0 = target
    i=0
    
    while kxi == kx0 and i<max_iter:
        eta_l = np.inf
        for k in index_list:
            if k != kx0:
                w_k = torch.autograd.grad(output[0][k], xi, retain_graph=True)[0] - torch.autograd.grad(output[0][kx0], xi, retain_graph=True)[0]
                f_k = output[0][k] - output[0][kx0]
                f_k.abs_()
                wk_norm_2 = np.linalg.norm(w_k.cpu().flatten(), ord=2)
                eta_l_k = f_k.item()/wk_norm_2
                if eta_l_k < eta_l:
                    eta_l = eta_l_k
                    ri = (eta_l_k/wk_norm_2)*w_k
        xi = xi + ri
        model.zero_grad()
        output = model(xi)
        _, kxi = torch.max(output, 1)
        i = i + 1
    original_output, output = original_output.detach(), output.detach()
    return original_output, output, xi.detach()

def clip(clean_image, per_image, eps):
	per_image = torch.max(clean_image-eps, torch.min(clean_image+eps, per_image))
	return per_image

