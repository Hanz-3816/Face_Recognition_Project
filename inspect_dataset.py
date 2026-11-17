import os
from pathlib import Path

dataset_path = Path("dataset/lfw-deepfunneled")

# Find all images recursively
images = list(dataset_path.rglob("*.jpg"))

# Identify individual folders
people = {
    image.parent
    for image in images
}

print("Dataset verification")
print("--------------------")

print("Total images:", len(images))
print("Number of individuals:", len(people))

print("\nSample image paths:")

for image in images[:5]:
    print(image)