# Face Verification Using Deep Learning

A comparative study of three pretrained deep learning models — **VGG-Face, FaceNet and ArcFace** — for face verification using the Labeled Faces in the Wild (LFW) dataset.

## 1. Project Overview

This project implements a face verification pipeline to evaluate and compare the performance of different deep learning-based face recognition models.

The objective is to determine whether two given face images belong to the same individual by extracting facial embeddings and measuring the distance between them.

Three pretrained models are evaluated:

- VGG-Face
- FaceNet
- ArcFace

The extracted embeddings are compared using Euclidean and Cosine distances. Their performance is evaluated using a 10-fold evaluation protocol and multiple classification metrics.

**DeepFace** is used as the framework for loading pretrained models and extracting facial embeddings.

## 2. Dataset

The project uses the **Labeled Faces in the Wild (LFW)** dataset, specifically its deep-funneled image collection.

The evaluation is performed using 6,000 image pairs:

| Pair type | Number of pairs |
|---|---:|
| Genuine (same identity) | 3,000 |
| Impostor (different identities) | 3,000 |
| **Total** | **6,000** |

The pairs are divided into 10 folds, with 600 pairs per fold. Each fold contains 300 genuine and 300 impostor pairs.

The original image archive is not included in this repository. It must be obtained separately from the [LFW dataset website](http://vis-www.cs.umass.edu/lfw/).

## 3. Methodology

The project follows these stages:

1. **Dataset Loading and Validation:** Load the evaluation pairs, construct image paths and verify that the required images are available.
2. **Embedding Extraction:** Use the three pretrained models through DeepFace to generate numerical representations (embeddings) of individual face images.
3. **Distance Calculation:** Calculate Euclidean and Cosine distances between the embeddings corresponding to each image pair.
4. **Threshold Selection:** Select a decision threshold using nine folds, with the remaining fold reserved for testing.
5. **Model Evaluation:** Evaluate the predictions on the held-out fold and aggregate the results across all ten folds.

A pair is classified as genuine when its distance is less than or equal to the selected threshold; otherwise, it is classified as an impostor.

## 4. Evaluation Metrics

The following metrics are used to assess model performance:

- Accuracy
- Precision
- Recall
- F1-score
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Equal Error Rate (EER)
- Receiver Operating Characteristic – Area Under the Curve (ROC-AUC)

The threshold is selected independently for each held-out fold using the other nine folds, preventing the test fold from influencing threshold selection.

## 5. Experimental Results

The following table presents the mean results across the ten evaluation folds. The best-performing distance metric for each model in this experiment is shown.

| Metric | VGG-Face (Euclidean) | FaceNet (Cosine) | ArcFace (Euclidean) |
|---|---:|---:|---:|
| Accuracy | 89.53% | 64.98% | 58.08% |
| Precision | 91.65% | 65.32% | 60.16% |
| Recall | 87.00% | 64.03% | 47.83% |
| F1-score | 89.26% | 64.62% | 53.27% |
| FAR | 7.93% | 34.07% | 31.67% |
| FRR | 13.00% | 35.97% | 52.17% |
| EER | 10.77% | 34.77% | 43.10% |
| ROC-AUC | 96.09% | 71.43% | 61.15% |

### Key Observations

- VGG-Face achieved the strongest overall performance in the current experiment, with 89.53% mean accuracy and 96.09% ROC-AUC.
- FaceNet performed better with Cosine distance than with Euclidean distance.
- ArcFace showed comparatively weaker discrimination between genuine and impostor pairs under the current experimental configuration.
- VGG-Face demonstrated relatively consistent performance across the ten evaluation folds.

These findings represent the results obtained with the current implementation and configuration. They should not be interpreted as a universal ranking of the underlying architectures.

## 6. Project Structure

```text
Face_Recognition_Project/
│
├── dataset/
│   ├── pairs.csv
│   ├── people.csv
│   ├── lfw_allnames.csv
│   ├── lfw_readme.csv
│   ├── matchpairsDevTest.csv
│   ├── matchpairsDevTrain.csv
│   ├── mismatchpairsDevTest.csv
│   ├── mismatchpairsDevTrain.csv
│   ├── peopleDevTest.csv
│   └── peopleDevTrain.csv
│
├── results/
│   ├── vgg_face_distances.csv
│   ├── facenet_distances.csv
│   ├── arcface_distances.csv
│   ├── evaluation_fold_results.csv
│   └── evaluation_summary.csv
│
├── dataset_loader.py
├── extract_embeddings.py
├── calculate_distances.py
├── evaluate_models.py
├── analyse.py
├── inspect_csv.py
├── inspect_dataset.py
├── inspect_main_pairs.py
├── test_embeddings.py
├── test_installation.py
│
├── README.md
└── .gitignore
```

## 7. Technologies Used

- Python
- DeepFace
- TensorFlow
- NumPy
- Pandas
- Pretrained deep learning models: VGG-Face, FaceNet and ArcFace

## 8. Limitations and Future Scope

The current experiments provide a comparative evaluation of three pretrained models on the selected LFW verification pairs.

Possible future improvements include:

- Investigating model-specific image preprocessing and normalization.
- Evaluating different face detection and alignment configurations.
- Studying the effect of different distance metrics and threshold-selection strategies.
- Extending the evaluation to additional datasets and more challenging real-world conditions.

## 9. References

1. Huang, G. B., Ramesh, M., Berg, T., and Learned-Miller, E. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*, University of Massachusetts, Amherst.
2. Serengil, S. I., and Ozpinar, A. *LightFace: A Hybrid Deep Face Recognition Framework*, 2020.
3. Parkhi, O. M., Vedaldi, A., and Zisserman, A. *Deep Face Recognition*, BMVC, 2015.
4. Schroff, F., Kalenichenko, D., and Philbin, J. *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR, 2015.
5. Deng, J., Guo, J., Xue, N., and Zafeiriou, S. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR, 2019.