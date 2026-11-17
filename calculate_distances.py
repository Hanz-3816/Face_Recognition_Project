from pathlib import Path
import numpy as np
import pandas as pd

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent
EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Model names and corresponding embedding files
MODELS = {
    "VGG-Face": "vgg_face_embeddings.npz",
    "Facenet": "facenet_embeddings.npz",
    "ArcFace": "arcface_embeddings.npz"
}


def load_embeddings(file_path):

    saved = np.load(file_path, allow_pickle=False)

    paths = saved["paths"]
    vectors = saved["embeddings"]

    # Create a mapping: image path -> embedding
    embedding_map = {
        str(path): vector
        for path, vector in zip(paths, vectors)
    }

    return embedding_map


def euclidean_distance(embedding1, embedding2):

    difference = embedding1 - embedding2

    return float(np.linalg.norm(difference))


def cosine_distance(embedding1, embedding2):

    dot_product = np.dot(embedding1, embedding2)

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot calculate cosine distance for a zero vector.")

    cosine_similarity = dot_product / (norm1 * norm2)

    # Protect against small floating-point errors
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

    return float(1.0 - cosine_similarity)


def calculate_model_distances(model_name, embedding_file, pairs):

    print(f"\nCalculating distances for {model_name}...")

    embedding_map = load_embeddings(embedding_file)

    # Check that every image in the evaluation pairs has an embedding
    required_paths = set(pairs["img1_path"]) | set(pairs["img2_path"])
    missing_paths = required_paths - set(embedding_map)

    if missing_paths:
        raise ValueError(
            f"{len(missing_paths)} images are missing embeddings for {model_name}."
        )

    results = []

    for index, row in pairs.iterrows():

        embedding1 = embedding_map[row["img1_path"]]
        embedding2 = embedding_map[row["img2_path"]]

        euclidean = euclidean_distance(embedding1, embedding2)
        cosine = cosine_distance(embedding1, embedding2)

        results.append({
            "pair_index": index,
            "fold": row["fold"],
            "label": row["label"],
            "pair_type": row["pair_type"],
            "euclidean_distance": euclidean,
            "cosine_distance": cosine
        })

    results_df = pd.DataFrame(results)

    # Save model-specific results
    output_file = RESULTS_DIR / (
        model_name.lower().replace("-", "_") + "_distances.csv"
    )

    results_df.to_csv(output_file, index=False)

    print("Total pairs processed:", len(results_df))
    print("Saved to:", output_file)

    print("\nAverage distances by pair type:")
    print(
        results_df.groupby("pair_type")[
            ["euclidean_distance", "cosine_distance"]
        ].mean()
    )

    return results_df


def main():

    from dataset_loader import load_lfw_pairs

    print("Loading LFW evaluation pairs...")

    pairs = load_lfw_pairs()

    print("Total pairs:", len(pairs))

    for model_name, filename in MODELS.items():

        embedding_file = EMBEDDINGS_DIR / filename

        if not embedding_file.exists():
            raise FileNotFoundError(
                f"Embedding file not found: {embedding_file}"
            )

        calculate_model_distances(
            model_name,
            embedding_file,
            pairs
        )

    print("\nDistance calculation completed for all models!")


if __name__ == "__main__":
    main()