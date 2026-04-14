import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize(IMAGE_SIZE)

    arr = torch.from_numpy(np.array(img)).float() / 255.0

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    arr = arr.permute(2, 0, 1)

    tensor = (arr - mean) / std
    return tensor.unsqueeze(0)