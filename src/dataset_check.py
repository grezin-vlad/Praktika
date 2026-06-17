import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = "data/EuroSAT"

classes = []
counts = []

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)

    if os.path.isdir(class_path):
        image_count = len(os.listdir(class_path))

        classes.append(class_name)
        counts.append(image_count)

df = pd.DataFrame({
    "Class": classes,
    "Images": counts
})

print("\nКлассы датасета:\n")
print(df)

print("\nВсего изображений:")
print(df["Images"].sum())

plt.figure(figsize=(12, 6))
plt.bar(classes, counts)

plt.xticks(rotation=45)
plt.title("Количество изображений по классам")
plt.xlabel("Класс")
plt.ylabel("Количество")

plt.tight_layout()

os.makedirs("results", exist_ok=True)

plt.savefig("results/class_distribution.png")

print("\nГрафик сохранён:")
print("results/class_distribution.png")