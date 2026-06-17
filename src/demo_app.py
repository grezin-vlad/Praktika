import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
import json
from datetime import datetime

# =========================
# КЛАССЫ EUROSAT
# =========================

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# =========================
# MODEL
# =========================

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    10
)

model.load_state_dict(
    torch.load(
        "models/efficientnet_b0.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# SAVE JSON
# =========================

def save_result(filename, predicted_class, confidence):

    result = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": filename,
        "class": predicted_class,
        "confidence": round(confidence, 4)
    }

    try:
        with open(
            "results/history.json",
            "r",
            encoding="utf-8"
        ) as f:

            history = json.load(f)

    except:

        history = []

    history.append(result)

    with open(
        "results/history.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4,
            ensure_ascii=False
        )

# =========================
# PREDICT
# =========================

def open_image():

    filepath = filedialog.askopenfilename(
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png")
        ]
    )

    if not filepath:
        return

    img = Image.open(filepath).convert("RGB")

    preview = img.resize((300, 300))

    preview = ImageTk.PhotoImage(preview)

    image_label.configure(image=preview)

    image_label.image = preview

    tensor = transform(img)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = CLASSES[
        predicted.item()
    ]

    conf = confidence.item() * 100

    result_label.config(
        text=f"Класс: {predicted_class}"
    )

    confidence_label.config(
        text=f"Уверенность: {conf:.2f}%"
    )

    save_result(
        filepath,
        predicted_class,
        conf
    )

# =========================
# GUI
# =========================

root = tk.Tk()

root.title(
    "EuroSAT Classification Demo"
)

root.geometry("600x500")

title = tk.Label(
    root,
    text="Классификация спутниковых снимков",
    font=("Arial", 16)
)

title.pack(pady=10)

btn = tk.Button(
    root,
    text="Выбрать изображение",
    command=open_image
)

btn.pack(pady=10)

image_label = tk.Label(root)

image_label.pack()

result_label = tk.Label(
    root,
    text="Класс: -",
    font=("Arial", 14)
)

result_label.pack(pady=10)

confidence_label = tk.Label(
    root,
    text="Уверенность: -",
    font=("Arial", 14)
)

confidence_label.pack()

root.mainloop()