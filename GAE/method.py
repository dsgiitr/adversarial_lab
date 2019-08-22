import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['GenAdv']

def GenAdv( model, image, epsilon, target, device):
	"""Generates Adv Examples for the Batch"""
	image, target = image.to(device), target.to(device)

	image.requires_grad = True
	original_output = model(image)

	loss = F.nll_loss(original_output, target)
	model.zero_grad()
	loss.backward()

	data_grad = image.grad.data

	perturbed_image = fgsm_attack(image, epsilon, data_grad)

	perturbed_output = model(perturbed_image)

	image.requires_grad = False
	original_output, perturbed_output = original_output.detach(), perturbed_output.detach()
	return original_output, perturbed_output, perturbed_image.detach()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image