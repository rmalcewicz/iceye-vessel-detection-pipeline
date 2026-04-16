import tifffile as tiff
import cv2
import numpy as np
import os
import gc

iceye_links = [
    "https://iceye-open-data-catalog.s3.amazonaws.com/data/spot/ICEYE_D1X6WB_20251107T033331Z_6934721_X50_SLH/ICEYE_D1X6WB_20251107T033331Z_6934721_X50_SLH_GRD.tif",
    "https://iceye-open-data-catalog.s3.amazonaws.com/data/spot/ICEYE_D1X6QK_20251107T033338Z_6934719_X50_SLH/ICEYE_D1X6QK_20251107T033338Z_6934719_X50_SLH_GRD.tif",
    "https://iceye-open-data-catalog.s3.amazonaws.com/data/spot/ICEYE_D1X6M9_20251107T033350Z_6934718_X50_SLH/ICEYE_D1X6M9_20251107T033350Z_6934718_X50_SLH_GRD.tif",
    "https://iceye-open-data-catalog.s3.amazonaws.com/data/spot/ICEYE_D1X6HY_20251107T033401Z_6934723_X50_SLH/ICEYE_D1X6HY_20251107T033401Z_6934723_X50_SLH_GRD.tif",
    "https://iceye-open-data-catalog.s3.amazonaws.com/data/spot/ICEYE_D1X6R1_20251107T033344Z_6934722_X50_SLH/ICEYE_D1X6R1_20251107T033344Z_6934722_X50_SLH_GRD.tif",
    "https://iceye-open-data-catalog.s3.amazonaws.com/data/spot/ICEYE_D1X6NJ_20251107T033356Z_6934717_X50_SLH/ICEYE_D1X6NJ_20251107T033356Z_6934717_X50_SLH_GRD.tif",
    "https://iceye-open-data-catalog.s3.amazonaws.com/data/spot/ICEYE_D1X6JD_20251107T033407Z_6934716_X50_SLH/ICEYE_D1X6JD_20251107T033407Z_6934716_X50_SLH_GRD.tif"
]

def download_and_process():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/test_set", exist_ok=True)

    for i, link in enumerate(iceye_links):
        raw_path = f"data/raw/iceye_{i}.tif"
        png_path = f"data/test_set/iceye_{i}.png"
        
        # Download if missing
        if not os.path.exists(raw_path):
            print(f"Downloading Image {i}...")
            os.system(f'wget -q "{link}" -O {raw_path}')
        
        # Load and Process
        print(f"Processing {raw_path}...")
        img = tiff.imread(raw_path).astype(np.float32)
        
        # Percentile Scaling
        # Prevents the "all black" output by ignoring outliers
        vmin, vmax = np.percentile(img, [2, 98])
        img_norm = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        # Normalize to 8-bit
        img_8bit = (img_norm * 255).astype(np.uint8)
        
        # Stack to 3 channels for YOLO compatibility
        img_3ch = np.stack([img_8bit, img_8bit, img_8bit], axis=-1)

        # Save
        cv2.imwrite(png_path, img_3ch)
        print(f"Saved: {png_path}")
        
        # Memory cleanup
        del img, img_norm, img_8bit, img_3ch
        gc.collect()

if __name__ == "__main__":
    download_and_process()
