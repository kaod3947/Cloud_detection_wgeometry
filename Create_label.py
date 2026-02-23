import os
import glob
import numpy as np
import rasterio
import cv2
from tqdm import tqdm


def parse_qa_pixel(qa):
    """
    Collection 2 QA_PIXEL
    """
    cloud = (
        ((qa & (1 << 1)) != 0) |  # dilated
        ((qa & (1 << 2)) != 0) |  # cirrus
        ((qa & (1 << 3)) != 0)    # cloud
    )

    shadow = (qa & (1 << 4)) != 0

    return cloud, shadow


def parse_bqa(bqa):
    """
        Landsat 8 Collection 1 BQA
        Pixel Value Interpretation 기반
        cloud / shadow / cirrus 분리
        """

    # --- Cloud (medium/high cloud 포함, cirrus 제외) ---
    cloud_values = set([2752, 2756, 2760, 2764, 2800, 2804, 2808, 2812, 3008, 3012, 3016, 3020])

    # --- Cloud Shadow ---
    shadow_values = set([2976, 2980, 2984, 2988, 3008, 3012, 3016, 3020])

    # --- Cirrus (cirrus 포함 pixel들) ---
    cirrus_values = set([6816, 6820, 6824, 6828, 6848, 6852, 6856, 6860, 6896, 6900, 6904, 6908,
        7072, 7076, 7080, 7084, 7104, 7108, 7112, 7116, 7840, 7844, 7848, 7852, 7872, 7876, 7880, 7884])

    cloud_mask = np.isin(bqa, list(cloud_values))
    shadow_mask = np.isin(bqa, list(shadow_values))
    cirrus_mask = np.isin(bqa, list(cirrus_values))

    return cloud_mask, shadow_mask, cirrus_mask


def process_landsat(root_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    scene_folders = [
        os.path.join(root_folder, d)
        for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ]

    for scene_path in tqdm(scene_folders, desc="Processing scenes"):

        scene_name = os.path.basename(scene_path)

        qa_pixel_file = glob.glob(os.path.join(scene_path, "*QA_PIXEL.TIF"))
        bqa_file = glob.glob(os.path.join(scene_path, "*BQA.TIF"))

        if qa_pixel_file:
            qa_path = qa_pixel_file[0]
            with rasterio.open(qa_path) as src:
                qa = src.read(1)
            cloud, shadow = parse_qa_pixel(qa)

        elif bqa_file:
            qa_path = bqa_file[0]
            with rasterio.open(qa_path) as src:
                qa = src.read(1)
            cloud, shadow, cirrus = parse_bqa(qa)

        else:
            print(f"No QA file found in {scene_name}")
            continue

        mask = np.zeros_like(qa, dtype=np.uint8)
        mask[cloud] = 1
        mask[shadow] = 2
        mask[cirrus] = 3

        save_path = os.path.join(output_folder, f"{scene_name}_label.png")
        cv2.imwrite(save_path, mask)

    print("✅ All scenes processed.")

if __name__ == "__main__":

    landsat_root = r"F:\01_data\00_Satellite\Landsat\LandSat8"   # Landsat 폴더 경로
    output_dir = r"F:\01_data\02_result\03_KST\Landsat_labels"  # 라벨 저장 폴더

    process_landsat(landsat_root, output_dir)
