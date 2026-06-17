import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# =====================
# DEVICE
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================
# DATASET
# =====================

DATASET_PATH = "data/EuroSAT"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

# =====================
# SPLIT DATA
# =====================

train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")

# =====================
# DATALOADERS
# =====================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# =====================
# MODEL
# =====================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model = model.to(device)

# =====================
# LOSS / OPTIMIZER
# =====================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =====================
# TRAINING
# =====================

EPOCHS = 5

train_losses = []
val_accuracies = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # =====================
    # VALIDATION
    # =====================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total

    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Loss: {train_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}")
    print("-" * 30)

# =====================
# SAVE MODEL
# =====================

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/resnet18.pth"
)

# =====================
# SAVE GRAPH
# =====================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    marker="o",
    label="Train Loss"
)

plt.plot(
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.title("ResNet18 Training")

plt.xlabel("Epoch")

plt.legend()

plt.grid(True)

plt.savefig(
    "results/resnet18_training.png"
)

plt.close()

print("Saved model + graph")

# =====================
# TEST ACCURACY
# =====================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total

print(f"\nTEST ACCURACY: {test_accuracy:.4f}")