import torch
from PIL import Image
from torchvision import transforms
import io
from src.model import SimpleCNN

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

classes = ["cat", "dog"]

def load_model():
    model = SimpleCNN()
   #model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
    model = torch.load("models/model.pt", map_location="cpu", weights_only=False)
    model.eval()
    return model

def predict(image_bytes, model):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return {
        "label": classes[pred.item()],
        "confidence": float(conf.item())
    }
