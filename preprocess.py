import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from image_preprocess import read_landsat_rgbn


class CloudDataset(Dataset):

    def __init__(self,
                 data_root,
                 cache_path,
                 raw_root,
                 label_root=None,
                 patch_size=512,
                 patch_stride=512,
                 satellite="landsat"):

        self.data_root = data_root
        self.cache_path = cache_path
        self.raw_root = raw_root
        self.label_root = label_root
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.satellite = satellite

        os.makedirs(cache_path, exist_ok=True)
        os.makedirs(os.path.join(cache_path, "img"), exist_ok=True)
        os.makedirs(os.path.join(cache_path, "label"), exist_ok=True)

        img_cache_dir = os.path.join(cache_path, "img")

        if len(os.listdir(img_cache_dir)) == 0:
            print("Building cache from raw data...")
            self._make_cache_from_raw()

        self.cache_files = sorted(os.listdir(img_cache_dir))

    # =========================
    # Scene 탐색
    # =========================
    def _find_all_scenes(self):

        scenes = []

        for scene_name in os.listdir(self.raw_root):
            scene_path = os.path.join(self.raw_root, scene_name)

            if not os.path.isdir(scene_path):
                continue

            scenes.append(scene_path)

        return scenes

    # =========================
    # 위성별 reader 선택
    # =========================
    def _read_scene(self, scene_path):

        if self.satellite == "landsat":
            return read_landsat_rgbn(scene_path)

        else:
            raise ValueError("Unsupported satellite type")

    # =========================
    # Cache 생성
    # =========================
    def _make_cache_from_raw(self):

        scenes = self._find_all_scenes()
        print("Found scenes:", len(scenes))

        img_dir = os.path.join(self.cache_path, "img")
        label_dir = os.path.join(self.cache_path, "label")

        for scene_path in scenes:

            img = self._read_scene(scene_path)
            if img is None:
                continue

            scene_name = os.path.basename(scene_path)

            full_label = self._read_label(scene_name)

            h, w, c = img.shape

            for y in range(0, h - self.patch_size + 1, self.patch_stride):
                for x in range(0, w - self.patch_size + 1, self.patch_stride):

                    img_patch = img[y:y + self.patch_size,
                                x:x + self.patch_size]

                    img_patch = torch.from_numpy(img_patch) \
                        .permute(2, 0, 1).float()

                    save_name = f"{scene_name}_{y}_{x}.pt"

                    # ✅ 이미지 저장
                    torch.save(
                        img_patch,
                        os.path.join(img_dir, save_name)
                    )

                    # ✅ 라벨 저장
                    if full_label is not None:
                        label_patch = full_label[
                                      y:y + self.patch_size,
                                      x:x + self.patch_size
                                      ]

                        label_patch = torch.from_numpy(label_patch).long()

                        torch.save(
                            label_patch,
                            os.path.join(label_dir, save_name)
                        )

        print("Cache build complete")
        print("Total cache files:",
              len(os.listdir(img_dir)))

    def save_to_cache(img, label, save_root, filename):
        img_dir = os.path.join(save_root, "img")
        label_dir = os.path.join(save_root, "label")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        img_path = os.path.join(img_dir, filename + ".png")
        label_path = os.path.join(label_dir, filename + ".png")

        cv2.imwrite(img_path, img)
        cv2.imwrite(label_path, label)
    # =========================
    # Label 읽기
    # =========================
    def _read_label(self, scene_name):

        if self.label_root is None:
            return None

        label_path = os.path.join(
            self.label_root,
            f"{scene_name}_label.png"
        )

        if not os.path.exists(label_path):
            print("Label not found:", label_path)
            return None

        label = cv2.imread(label_path, 0)
        return label

    # =========================
    # 필수 함수
    # =========================
    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):

        file = self.cache_files[idx]

        img_path = os.path.join(self.cache_path, "img", file)
        label_path = os.path.join(self.cache_path, "label", file)

        img = torch.load(img_path)

        if os.path.exists(label_path):
            label = torch.load(label_path)
        else:
            label = torch.zeros(
                (self.patch_size, self.patch_size)
            ).long()

        return img, label