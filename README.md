#  CIFAR-10 Image Classification using K-Nearest Neighbors (KNN)

This project actually implements **image classification** on the **CIFAR-10 dataset** using a **custom-built K-Nearest Neighbors (KNN)** algorithm written from scratch, no external ML libraries like `scikit-learn` are used for the core logic.

The focus is on understanding how KNN works under the hood, from computing distances to predicting labels, while applying it on a real-world image dataset.

-

## 📘 Project Overview

The CIFAR-10 dataset contains **60,000 color images** across **10 classes**, such as:

> airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

Each image is **32×32×3** pixels (RGB).

Your implementation:

* Loads the dataset
* Flattens and normalizes the images
* Uses **Euclidean distance** for similarity
* Classifies images based on the **k-nearest neighbors**

-

## ⚙️ Key Functions

### 🧮 `compute_distances(X_train, X_test)`

Efficiently computes the pairwise Euclidean distance between each test sample and all training samples using vectorized NumPy operations:

[
d(x, y) = \sqrt{(x^2 + y^2 - 2xy)}
]

**Steps:**

1. Compute squared sums of test and train samples
2. Use matrix multiplication for the cross term
3. Combine and take square root for final distances

-

### 🔍 `predict_labels(dists, y_train, k=5)`

Predicts class labels for each test image by:

1. Finding the `k` nearest neighbors (lowest distances)
2. Retrieving their labels from the training set
3. Choosing the most frequent label using `np.bincount().argmax()`

-

## 🧠 Algorithm Flow

1. **Load CIFAR-10** dataset
2. **Preprocess** data

   * Flatten images (32×32×3 → 3072)
   * Normalize pixel values (0–1)
   * Split into train and test sets
3. **Compute distances** between test and train samples using `compute_distances()`
4. **Predict labels** using `predict_labels()`
5. **Evaluate accuracy**

-

## 📈 Example Accuracy

| k  | Accuracy |
| -- | -------- |
| 1  | ~32%     |
| 5  | ~34%     |
| 10 | ~33%     |

> Note: KNN on CIFAR-10 is mainly educational — deep learning methods (CNNs) perform much better on this dataset.

-

## 🧩 Dependencies

You only need **NumPy** and optionally **Matplotlib** for visualization.

```bash
pip install numpy matplotlib
```

-

## 🚀 Run the Project

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/cifar10-knn.git
   cd cifar10-knn
   ```

2. Run the script:

   ```bash
   python cifar10_knn.py
   ```

3. Display sample predictions:

   ```python
   import matplotlib.pyplot as plt
   # visualize random samples and predictions
   ```


## Author

UNIQUE
