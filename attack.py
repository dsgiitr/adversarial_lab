import json

from commons import get_model, transform_image, rev_transform

import GAE
import torch

model = get_model()
imagenet_class_index = json.load(open('imagenet_class_index.json'))


def fgsm_untargeted(image_bytes, epsilon, device=torch.device('cpu')):
	try:
		tensor = transform_image(image_bytes=image_bytes)
		#outputs = model.forward(tensor)
		original_output, perturbed_output, perturbed_image = GAE.fgsm(model, tensor, epsilon, device)
	except Exception:
	    return 0, 'error'
	or_percentage = torch.nn.functional.softmax(original_output, dim=1)[0] * 100
	_, or_indices = torch.sort(original_output, descending=True)
	per_percentage = torch.nn.functional.softmax(perturbed_output, dim=1)[0] * 100
	_, per_indices = torch.sort(perturbed_output, descending=True)
	top5_original = [(imagenet_class_index[str(idx.item())][1], or_percentage[idx.item()].item()) for idx in or_indices[0][:5]]
	top5_perturbed = [(imagenet_class_index[str(idx.item())][1], or_percentage[idx.item()].item()) for idx in per_indices[0][:5]]
	perturbed_image = rev_transform(perturbed_image[0])

	return dict(top5_original), dict(top5_perturbed), perturbed_image