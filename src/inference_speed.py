import time
import torch
import torch.nn as nn

from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dummy = torch.randn(
    1,
    3,
    224,
    224
).to(device)

MODELS = {}

# ResNet18
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet18.pth", map_location=device))
MODELS["ResNet18"] = model

# ResNet50
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet50.pth", map_location=device))
MODELS["ResNet50"] = model

# MobileNetV3
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    10
)
model.load_state_dict(torch.load("models/mobilenetv3.pth", map_location=device))
MODELS["MobileNetV3"] = model

# EfficientNet
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    10
)
model.load_state_dict(torch.load("models/efficientnet_b0.pth", map_location=device))
MODELS["EfficientNet-B0"] = model

# DenseNet121
model = models.densenet121(weights=None)
model.classifier = nn.Linear(
    model.classifier.in_features,
    10
)
model.load_state_dict(torch.load("models/densenet121.pth", map_location=device))
MODELS["DenseNet121"] = model

print("\nINFERENCE SPEED")
print("=" * 50)

for name, model in MODELS.items():

    model = model.to(device)
    model.eval()

    # прогрев
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    start = time.time()

    with torch.no_grad():
        for _ in range(100):
            model(dummy)

    end = time.time()

    avg_ms = ((end - start) / 100) * 1000

    print(f"{name:<20} {avg_ms:.2f} ms/image")