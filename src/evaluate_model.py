import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================
# ВЫБОР МОДЕЛИ
# =====================

MODEL_NAME = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Model :", MODEL_NAME)

# =====================
# DATASET
# =====================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    "data/EuroSAT",
    transform=transform
)

train_size = int(len(dataset) * 0.70)
val_size = int(len(dataset) * 0.15)
test_size = len(dataset) - train_size - val_size

_, _, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# =====================
# LOAD MODEL
# =====================

if MODEL_NAME == "resnet18":

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    weights_path = "models/resnet18.pth"

elif MODEL_NAME == "resnet50":

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    weights_path = "models/resnet50.pth"

elif MODEL_NAME == "mobilenetv3":

    model = models.mobilenet_v3_large(weights=None)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        10
    )

    weights_path = "models/mobilenetv3.pth"

elif MODEL_NAME == "efficientnet":

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        10
    )

    weights_path = "models/efficientnet_b0.pth"

elif MODEL_NAME == "densenet121":

    model = models.densenet121(weights=None)

    model.classifier = nn.Linear(
        model.classifier.in_features,
        10
    )

    weights_path = "models/densenet121.pth"

else:

    raise Exception("Unknown model")

model.load_state_dict(
    torch.load(
        weights_path,
        map_location=device
    )
)

model = model.to(device)
model.eval()

# =====================
# PREDICT
# =====================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predicted.cpu().numpy()
        )

# =====================
# METRICS
# =====================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print("\nRESULTS")
print("=" * 40)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# =====================
# CONFUSION MATRIX
# =====================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

os.makedirs(
    "results",
    exist_ok=True
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.classes
)

disp.plot(
    xticks_rotation=90
)

plt.tight_layout()

plt.savefig(
    f"results/confusion_matrix_{MODEL_NAME}.png"
)

plt.close()

print(
    f"\nConfusion matrix saved: results/confusion_matrix_{MODEL_NAME}.png"
)