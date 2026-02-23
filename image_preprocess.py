# image_preprocess.py

import os
import numpy as np
from osgeo import gdal


def read_landsat_rgbn(scene_path):

    def read_band(band_name):
        for f in os.listdir(scene_path):
            if band_name in f and f.lower().endswith(".tif"):
                ds = gdal.Open(os.path.join(scene_path, f))
                return ds.GetRasterBand(1).ReadAsArray()
        return None

    blue = read_band("B2")
    green = read_band("B3")
    red = read_band("B4")
    nir = read_band("B5")

    if any(b is None for b in [blue, green, red, nir]):
        print("Missing band in:", scene_path)
        return None

    img = np.stack([red, green, blue, nir], axis=-1).astype(np.float32)
    img /= 10000.0

    return img


# 나중에 추가 가능
def read_sentinel_rgbn(scene_path):
    pass


def read_kompsat_rgbn(scene_path):
    pass
