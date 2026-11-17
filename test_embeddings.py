from deepface import DeepFace
from dataset_loader import load_lfw_pairs
import numpy as np

# Load our prepared LFW pairs
data = load_lfw_pairs()

# Select the first image from the first pair
image_path = data.iloc[0]["img1_path"]

# Models we want to evaluate
models = ["VGG-Face", "Facenet", "ArcFace"]

print("Testing face embedding extraction...")
print("Image:", image_path)

for model_name in models:

    print(f"\nProcessing with {model_name}...")

    # Generate face embedding
    result = DeepFace.represent(
        img_path=image_path,
        model_name=model_name,
        detector_backend="skip"
    )

    # Extract embedding vector
    embedding = np.array(result[0]["embedding"])

    print("Embedding shape:", embedding.shape)
    print("First 5 values:", embedding[:5])
    print("Vector magnitude:", np.linalg.norm(embedding))

print("\nAll three models tested successfully!")