"""
traditional_ml.py  (Role A)
Extracts hand-crafted features (colour histogram, Hu moments, Haralick-style texture)
and trains NB, LDA, KNN, RF, SVM. Produces:
  results_table.csv   - test accuracy for every feature combination
  cv_boxplot.png      - 10-fold CV accuracy box plot (all features)
Usage:
  python traditional_ml.py --data_dir /kaggle/working/data
"""
import argparse
import os
from itertools import combinations

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.feature import graycomatrix, graycoprops
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def features(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # His: 8x8x8 HSV colour histogram
    his = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    his = cv2.normalize(his, his).flatten()
    # Hu: 7 shape moments (log-scaled)
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    # Ha: GLCM texture statistics (gray quantised to 32 levels)
    q = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    ha = np.concatenate([graycoprops(glcm, p).flatten() for p in props])
    # Raw: tiny 32x32 pixels (the "without feature engineering" baseline)
    raw = cv2.resize(img, (32, 32)).flatten() / 255.0
    return his, ha, hu, raw


def load(data_dir, split):
    df = pd.read_csv(os.path.join(data_dir, f"labels_{split}.csv"))
    paths = [os.path.join(data_dir, split, f) for f in df["image"]]
    out = Parallel(n_jobs=-1)(delayed(features)(p) for p in paths)
    f = {k: np.array([o[i] for o in out]) for i, k in enumerate(["His", "Ha", "Hu", "Raw"])}
    return f, df["label"].values


def models():
    return {
        "NB": make_pipeline(StandardScaler(), GaussianNB()),
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "RF": RandomForestClassifier(n_estimators=70, random_state=0, n_jobs=-1),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    }


def stack(f, names):
    return np.hstack([f[n] for n in names])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    a = p.parse_args()

    print("Extracting features (a few minutes)...")
    ftr, ytr = load(a.data_dir, "train")
    fte, yte = load(a.data_dir, "test")
    print("Blind 'no ship' accuracy on test:", round(1 - yte.mean(), 3))

    combos = {"His": ["His"], "Ha": ["Ha"], "Hu": ["Hu"],
              "His+Ha": ["His", "Ha"], "Ha+Hu": ["Ha", "Hu"], "Hu+His": ["Hu", "His"],
              "w/ all": ["His", "Ha", "Hu"], "w/o (raw)": ["Raw"]}
    rows = []
    for mname in models():
        row = {"Alg": mname}
        for cname, names in combos.items():
            m = models()[mname]
            m.fit(stack(ftr, names), ytr)
            row[cname] = round(m.score(stack(fte, names), yte), 3)
        rows.append(row)
        print(row)
    table = pd.DataFrame(rows)
    table.to_csv("results_table.csv", index=False)
    print(table.to_string(index=False))

    # 10-fold CV with all features
    X, cv_scores = stack(ftr, ["His", "Ha", "Hu"]), {}
    skf = StratifiedKFold(10, shuffle=True, random_state=0)
    for mname in models():
        cv_scores[mname] = cross_val_score(models()[mname], X, ytr, cv=skf)
        print(mname, "CV mean:", cv_scores[mname].mean().round(3))
    plt.boxplot(list(cv_scores.values()), labels=list(cv_scores.keys()))
    plt.ylabel("Accuracy")
    plt.title("10-fold CV accuracy (all features)")
    plt.savefig("cv_boxplot.png", dpi=150, bbox_inches="tight")
    print("Saved results_table.csv and cv_boxplot.png")


if __name__ == "__main__":
    main()
