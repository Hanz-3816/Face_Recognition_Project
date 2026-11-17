import numpy as np
from pathlib import Path

embedding_dir = Path("embeddings")

models = {
    "VGG-Face": "vgg_face_embeddings.npz",
    "FaceNet": "facenet_embeddings.npz",
    "ArcFace": "arcface_embeddings.npz"
}

for model, filename in models.items():
    data = np.load(embedding_dir / filename)

    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1)

    print(f"\n{model}")
    print("Embedding shape:", embeddings.shape)
    print("Minimum norm:", norms.min())
    print("Maximum norm:", norms.max())
    print("Mean norm:", norms.mean())
    print("Standard deviation:", norms.std())
    print("Zero vectors:", np.sum(norms == 0))
    print("Non-finite values:", np.sum(~np.isfinite(embeddings)))