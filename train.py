import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from preprocess import CloudDataset
from model import build_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = CloudDataset(
        data_root="data",
        cache_path="cache",
        raw_root=r"F:\01_data\00_Satellite\Landsat\LandSat8",
        label_root=r"F:\01_data\02_result\03_KST\Landsat_labels",
        patch_size=512,
        patch_stride=512
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    for imgs, labels in train_loader:
        print(imgs.shape, labels.shape)
        break

    print('test complete~')


    # -----------------------
    # Model
    # -----------------------
    model = build_model(num_classes=4)
    model = model.to(device)

    # -----------------------
    # Loss & Optimizer
    # -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20

    # -----------------------
    # Training Loop
    # -----------------------
    for epoch in range(num_epochs):

        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader)

        for imgs, labels in loop:

            imgs = imgs.to(device)           # [B,4,H,W]
            labels = labels.to(device)       # [B,H,W]

            optimizer.zero_grad()

            outputs = model(imgs)['out']     # 🔥 DeepLab 출력

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.4f}")

        torch.save(model.state_dict(), f"deeplab_epoch_{epoch+1}.pth")

    print("Training Complete")
if __name__ == "__main__":
    main()