"""
prepare_data.py  (Role A)
Samples a random subset of the Airbus Ship Detection images, labels them
(1 = has ship, 0 = no ship), resizes to 256x256 and splits into train/dev/test.

Run inside a Kaggle Notebook with the competition data attached:
  python prepare_data.py --raw_dir /kaggle/input/airbus-ship-detection \
                         --out_dir /kaggle/working/data

Output layout (share this with Role B):
  data/train/*.jpg  data/dev/*.jpg  data/test/*.jpg
  data/labels_train.csv  data/labels_dev.csv  data/labels_test.csv   (columns: image,label)
"""
import argparse
import os

import cv2
import pandas as pd
from tqdm import tqdm

IMG = 256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="/kaggle/input/airbus-ship-detection")
    p.add_argument("--out_dir", default="data")
    p.add_argument("--n_train", type=int, default=6000)
    p.add_argument("--n_dev", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    # 1. Labels: an image with no EncodedPixels has no ships
    csv_path = os.path.join(a.raw_dir, "train_ship_segmentations_v2.csv")
    df = pd.read_csv(csv_path)
    labels = (df.groupby("ImageId")["EncodedPixels"]
                .apply(lambda s: int(s.notna().any()))
                .reset_index())
    labels.columns = ["image", "label"]
    print("Total images:", len(labels), "| ship fraction:", round(labels.label.mean(), 3))

    # 2. Uniform random sample (keeps the natural class imbalance)
    total = a.n_train + a.n_dev + a.n_test
    sample = labels.sample(n=total, random_state=a.seed).reset_index(drop=True)
    splits = {
        "train": sample.iloc[: a.n_train],
        "dev": sample.iloc[a.n_train : a.n_train + a.n_dev],
        "test": sample.iloc[a.n_train + a.n_dev :],
    }

    # 3. Resize + save
    img_dir = os.path.join(a.raw_dir, "train_v2")
    for name, part in splits.items():
        out = os.path.join(a.out_dir, name)
        os.makedirs(out, exist_ok=True)
        for fname in tqdm(part["image"], desc=name):
            img = cv2.imread(os.path.join(img_dir, fname))
            img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out, fname), img)
        part.to_csv(os.path.join(a.out_dir, f"labels_{name}.csv"), index=False)
        print(name, len(part), "images | ship fraction:", round(part.label.mean(), 3))


if __name__ == "__main__":
    main()
