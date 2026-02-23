import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

from preprocess import CloudDataset
from model import build_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Dataset (cache 기반)
    # -----------------------
    dataset = CloudDataset(
        data_root="data",
        cache_path="cache",
        raw_root=r"F:\01_data\00_Satellite\Landsat\LandSat8",
        label_root=None,  # 예측만 할 거라 필요 없음
        patch_size=512,
        patch_stride=512
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # -----------------------
    # Model
    # -----------------------
    model = build_model(num_classes=4)
    model.load_state_dict(torch.load("deeplab_epoch_20.pth", map_location=device))
    model = model.to(device)
    model.eval()

    os.makedirs("predictions", exist_ok=True)

    # -----------------------
    # Inference Loop
    # -----------------------
    with torch.no_grad():

        for i, (imgs, _) in enumerate(loader):

            imgs = imgs.to(device)  # [1,4,H,W]

            outputs = model(imgs)["out"]  # [1,C,H,W]

            preds = torch.argmax(outputs, dim=1)  # [1,H,W]

            pred_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)

            save_path = os.path.join("predictions", f"pred_{i}.png")
            cv2.imwrite(save_path, pred_np)

            print(f"Saved: {save_path}")

    print("Prediction Complete")


if __name__ == "__main__":
    main()
