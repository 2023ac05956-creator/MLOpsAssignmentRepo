from src.data_preprocessing import get_transforms
from PIL import Image
import torch

def test_resize():
    transform = get_transforms(train=False)
    img = Image.new("RGB", (500, 500))
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)
