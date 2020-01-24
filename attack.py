import json

from commons import get_model, transform_image, rev_transform

import GAE
import torch

imagenet_class_index = json.load(open('imagenet_class_index.json'))


def fgsm_untargeted( model_inp, image_bytes, epsilon, device=torch.device('cpu')):
	model = get_model(model_inp)
	tensor = transform_image(image_bytes=image_bytes)
	#outputs = model.forward(tensor)
	original_output, perturbed_output, perturbed_image = GAE.fgsm(model, tensor, epsilon, device)
	cpudev = torch.device('cpu')
	original_output, perturbed_output = original_output.to(cpudev), perturbed_output.to(cpudev)
	perturbed_image, tensor = perturbed_image.to(cpudev), tensor.to(cpudev)
	per_tensor = perturbed_image - tensor
	or_percentage = torch.nn.functional.softmax(original_output, dim=1)[0] * 100
	_, or_indices = torch.sort(original_output, descending=True)
	per_percentage = torch.nn.functional.softmax(perturbed_output, dim=1)[0] * 100
	_, per_indices = torch.sort(perturbed_output, descending=True)
	top5_original = [(imagenet_class_index[str(idx.item())][1], or_percentage[idx.item()].item()) for idx in or_indices[0][:5]]
	top5_perturbed = [(imagenet_class_index[str(idx.item())][1],or_percentage[idx.item()].item()) for idx in per_indices[0][:5]]
	perturbed_image = rev_transform(perturbed_image[0])
	original_image = rev_transform(tensor[0])
	perturbation = rev_transform(per_tensor[0])
	del model
	return dict(top5_original), dict(top5_perturbed), perturbed_image, original_image, perturbation

def fgsm_targeted(model, image_bytes, epsilon, target, device=torch.device('cpu')):
	tensor = transform_image(image_bytes=image_bytes)
	#outputs = model.forward(tensor)
	target = torch.tensor([target])
	original_output, perturbed_output, perturbed_image = GAE.fgsm_targeted(model, tensor, epsilon, target, device)
	cpudev = torch.device('cpu')
	original_output, perturbed_output = original_output.to(cpudev), perturbed_output.to(cpudev)
	perturbed_image, tensor = perturbed_image.to(cpudev), tensor.to(cpudev)
	per_tensor = perturbed_image - tensor
	or_percentage = torch.nn.functional.softmax(original_output, dim=1)[0] * 100
	_, or_indices = torch.sort(original_output, descending=True)
	per_percentage = torch.nn.functional.softmax(perturbed_output, dim=1)[0] * 100
	_, per_indices = torch.sort(perturbed_output, descending=True)
	top5_original = [(imagenet_class_index[str(idx.item())][1], or_percentage[idx.item()].item()) for idx in or_indices[0][:5]]
	top5_perturbed = [(imagenet_class_index[str(idx.item())][1], or_percentage[idx.item()].item()) for idx in per_indices[0][:5]]
	perturbed_image = rev_transform(perturbed_image[0])
	original_image = rev_transform(tensor[0])
	perturbation = rev_transform(per_tensor[0])
	return dict(top5_original), dict(top5_perturbed), perturbed_image, original_image, perturbation

