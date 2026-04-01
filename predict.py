import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
from caption import generate_caption

# Load pretrained model
model = models.resnet50(pretrained=True)
model.eval()

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = urllib.request.urlopen(LABELS_URL).read().decode("utf-8").splitlines()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)

    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top3 = torch.topk(probs, 3)

    labels = [classes[i] for i in top3.indices.tolist()]
    return labels

# TEST
if __name__ == "__main__":
    result = predict_image("test.jpg")
    print("Predicted labels:", result)

labels = predict_image("test.jpg")
print("Predicted labels:", labels)

caption = generate_caption("test.jpg")
print("Caption:", caption)