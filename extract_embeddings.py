from pathlib import Path
import numpy as np
from deepface import DeepFace
from dataset_loader import load_lfw_pairs

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)

# Models to evaluate
MODELS = {
    "VGG-Face": "vgg_face_embeddings.npz",
    "Facenet": "facenet_embeddings.npz",
    "ArcFace": "arcface_embeddings.npz"
}

# Save a checkpoint after this many new images
CHECKPOINT_INTERVAL = 100


def save_embeddings(file_path, embeddings):

    paths = list(embeddings.keys())

    vectors = np.array(
        [embeddings[path] for path in paths],
        dtype=np.float32
    )

    np.savez_compressed(
        file_path,
        paths=np.array(paths),
        embeddings=vectors
    )


def load_existing_embeddings(file_path):

    if not file_path.exists():
        return {}

    saved = np.load(file_path, allow_pickle=False)

    return {
        str(path): vector
        for path, vector in zip(
            saved["paths"],
            saved["embeddings"]
        )
    }


def extract_model_embeddings(model_name, output_file, image_paths):

    print(f"\n{'=' * 55}")
    print(f"Starting model: {model_name}")
    print(f"{'=' * 55}")

    # Load previously saved embeddings, if available
    embeddings = load_existing_embeddings(output_file)

    print("Already extracted:", len(embeddings))

    remaining_paths = [
        path for path in image_paths
        if path not in embeddings
    ]

    print("Images remaining:", len(remaining_paths))

    failed_images = []
    new_count = 0

    for index, image_path in enumerate(remaining_paths, start=1):

        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=model_name,
                detector_backend="skip"
            )

            embedding = np.array(
                result[0]["embedding"],
                dtype=np.float32
            )

            embeddings[image_path] = embedding

            new_count += 1

        except Exception as error:

            print(f"\nFailed: {image_path}")
            print("Reason:", error)

            failed_images.append(image_path)

        # Display progress
        if index % 25 == 0 or index == len(remaining_paths):
            print(
                f"Processed {index}/{len(remaining_paths)} "
                f"| Total saved: {len(embeddings)}"
            )

        # Save periodic checkpoints
        if new_count > 0 and new_count % CHECKPOINT_INTERVAL == 0:
            save_embeddings(output_file, embeddings)
            print("Checkpoint saved.")

    # Final save
    save_embeddings(output_file, embeddings)

    # Save failed image paths separately
    if failed_images:
        failed_file = OUTPUT_DIR / (
            model_name.lower().replace("-", "_") + "_failed.txt"
        )

        failed_file.write_text("\n".join(failed_images))

    print(f"\nCompleted: {model_name}")
    print("Successfully extracted:", len(embeddings))
    print("Failed images:", len(failed_images))
    print("Saved to:", output_file)


def main():

    print("Loading LFW evaluation pairs...")

    data = load_lfw_pairs()

    # Collect unique image paths from both columns
    image_paths = list(dict.fromkeys(
        data["img1_path"].tolist() +
        data["img2_path"].tolist()
    ))

    print("Total evaluation pairs:", len(data))
    print("Unique images:", len(image_paths))

    # Verify that every image exists
    missing = [
        path for path in image_paths
        if not Path(path).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image files are missing."
        )

    print("All image paths verified.")

    # Extract embeddings model by model
    for model_name, filename in MODELS.items():

        output_file = OUTPUT_DIR / filename

        extract_model_embeddings(
            model_name,
            output_file,
            image_paths
        )

    print("\nAll model extraction jobs finished!")


if __name__ == "__main__":
    main()