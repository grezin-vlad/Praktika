from torchvision import datasets
from torch.utils.data import random_split

DATASET_PATH = "data/EuroSAT"

dataset = datasets.ImageFolder(DATASET_PATH)

total_size = len(dataset)

train_size = int(total_size * 0.70)
val_size = int(total_size * 0.15)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

print(f"Всего изображений: {total_size}")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")

print("\nКлассы:")

for idx, class_name in enumerate(dataset.classes):
    print(idx, class_name)