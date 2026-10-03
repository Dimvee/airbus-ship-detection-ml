# Airbus Ship Detection: Traditional ML vs CNN

UE24CS352A Machine Learning Mini-Project

**Team:**
- Dimpal N (PES2UG24CS159)
- Meenakshi P (PES2UG25CS814)

## Problem statement
Classify satellite images as **ship / no ship** (binary classification) using the Airbus Ship Detection Challenge data. We compare traditional machine learning on hand-crafted features (Naive Bayes, LDA, KNN, Random Forest, SVM) with deep learning (a simple CNN trained from scratch and a transfer-learning CNN).

## Dataset
- Source: [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection) on Kaggle (about 192k images, 768x768; about 22% contain ships).
- Subset used: 10,000 images sampled uniformly at random (seed 42): 6,000 train, 2,000 dev, 2,000 test.
- Images resized to 256x256. An image with no entry in the segmentation CSV is labelled "no ship".
- The data is **not** stored in this repo. Generate it with `prepare_data.py` (see below).

## Repository structure
```
prepare_data.py        sample, label, resize and split the data
traditional_ml.py      feature extraction + NB, LDA, KNN, RF, SVM
results_table.csv      test accuracy table (traditional ML)
cv_boxplot.png         10-fold CV box plot
[CNN scripts]          [teammate: add file names here]
requirements.txt
```

## Setup
```
pip install -r requirements.txt
```
We ran the data preparation and traditional ML in a Kaggle Notebook (the full dataset is over 30 GB, so it is not downloaded locally).

## How to run

**1. Prepare the data** (Kaggle Notebook with the Airbus competition data attached):
```
python prepare_data.py --raw_dir /kaggle/input/competitions/airbus-ship-detection --out_dir /kaggle/working/data
```
This creates `data/train`, `data/dev`, `data/test` and `labels_train.csv`, `labels_dev.csv`, `labels_test.csv` (columns: `image,label`).

**2. Traditional ML:**
```
python traditional_ml.py --data_dir /kaggle/working/data
```
This extracts features, trains the 5 models, and writes `results_table.csv` and `cv_boxplot.png`.

**3. CNN models:** [teammate: add run instructions here]

## Methods (traditional ML)
- Features: HSV colour histogram (His), Hu moments (Hu), GLCM texture statistics (Ha); tested alone and combined, and compared with raw pixels.
- Models: Gaussian Naive Bayes, LDA, KNN (k=5), Random Forest (70 trees), SVM (RBF kernel). Features are standardised for all models except Random Forest.
- Evaluation: test accuracy on the held-out test set, plus 10-fold cross-validation on the training set.

## Results (traditional ML, test accuracy)
Always predicting "no ship" gives 0.784 on our test set.

| Alg | His | Ha | Hu | His+Ha | Ha+Hu | Hu+His | All | Raw pixels |
|---|---|---|---|---|---|---|---|---|
| NB | 0.794 | 0.692 | 0.784 | 0.795 | 0.671 | 0.795 | 0.796 | 0.520 |
| LDA | 0.793 | 0.785 | 0.784 | 0.804 | 0.782 | 0.798 | 0.805 | 0.698 |
| KNN | 0.809 | 0.804 | 0.733 | 0.820 | 0.787 | 0.804 | 0.821 | 0.767 |
| RF | 0.901 | 0.855 | 0.773 | 0.902 | 0.858 | 0.898 | **0.903** | 0.811 |
| SVM | 0.804 | 0.793 | 0.784 | 0.818 | 0.811 | 0.805 | 0.820 | 0.786 |

10-fold CV mean accuracy (all features): NB 0.777, LDA 0.807, KNN 0.825, **RF 0.912**, SVM 0.816.

Random Forest is the best traditional method, and feature engineering improves every model compared with raw pixels.

**CNN results:** [teammate: add results here]

## Reference
Based on the approach in "Airbus Ship Detection: Traditional vs. Convolutional Neural Network Approach" (Chen, Zheng, Zhou, CS229 project).
