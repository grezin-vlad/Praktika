from pathlib import Path

models = [
    "resnet18.pth",
    "resnet50.pth",
    "mobilenetv3.pth",
    "efficientnet_b0.pth",
    "densenet121.pth"
]

print("\nMODEL SIZES")
print("=" * 50)

for model in models:

    path = Path("models") / model

    size_mb = path.stat().st_size / (1024 * 1024)

    print(
        f"{model:<20} {size_mb:.2f} MB"
    )