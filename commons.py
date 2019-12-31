import io


from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import torch

def get_model():
    model = models.alexnet(pretrained=True)
    model.eval()
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def rev_transform(image):
	inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
	)
	image = inv_normalize(image)
	npimg = image.cpu().numpy()
	npimg = np.transpose(npimg, (1, 2, 0))
	return npimg


# ImageNet classes are often of the form `can_opener` or `Egyptian_cat`
# will use this method to properly format it so that we get
# `Can Opener` or `Egyptian Cat`
def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name