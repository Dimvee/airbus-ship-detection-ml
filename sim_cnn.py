import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ShipDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = root
        self.split = split
        self.df = pd.read_csv(os.path.join(root, f"labels_{split}.csv"))

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root, self.split, row["image"])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image, label


class SIMCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="/content/sim_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIMCNN().to(device)
    print("Device:", device)

    if args.dummy:
        with torch.no_grad():
            output = model(torch.randn(2, 3, 256, 256).to(device))
        print("Dummy test passed. Output shape:", tuple(output.shape))
        return

    if not args.data_dir:
        parser.error("--data_dir is required unless --dummy is used")

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds = ShipDataset(args.data_dir, "train", augment=True)
    dev_ds = ShipDataset(args.data_dir, "dev")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"train_loss": [], "dev_loss": [],
               "train_acc": [], "dev_acc": []}
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, count = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            probs = model(images)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += ((probs >= 0.5) == (labels >= 0.5)).sum().item()
            count += len(labels)

        history["train_loss"].append(total_loss / count)
        history["train_acc"].append(correct / count)

        model.eval()
        total_loss, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in dev_loader:
                images, labels = images.to(device), labels.to(device)
                probs = model(images)
                loss = criterion(probs, labels)
                total_loss += loss.item() * len(labels)
                correct += ((probs >= 0.5) == (labels >= 0.5)).sum().item()
                count += len(labels)

        dev_loss = total_loss / count
        history["dev_loss"].append(dev_loss)
        history["dev_acc"].append(correct / count)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train loss: {history['train_loss'][-1]:.4f} | "
              f"Dev loss: {dev_loss:.4f} | "
              f"Train acc: {history['train_acc'][-1]:.4f} | "
              f"Dev acc: {history['dev_acc'][-1]:.4f}", flush=True)

        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "sim_cnn_best.pth"))

    pd.DataFrame(history).to_csv(
        os.path.join(args.output_dir, "training_history.csv"), index=False)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["dev_loss"], label="Dev")
    plt.title("SIM-CNN Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["dev_acc"], label="Dev")
    plt.title("SIM-CNN Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "sim_cnn_curves.png"))
    plt.show()
    print("Training results saved in", args.output_dir)


if __name__ == "__main__":
    main()
