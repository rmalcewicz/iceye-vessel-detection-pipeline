import cv2
import numpy as np
import os
import gc
from sahi.utils.cv import visualize_object_predictions
from src.strategy_trad import detect_ships_traditional
from src.strategy_pretrained import detect_pretrained
from src.strategy_expert import detect_expert

def export_panels_for_index(index):

    path = f'data/test_set/iceye_{index}.png'
    save_dir = f'output/detections/iceye_{index}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load base image
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"Skipping index {index}: File not found.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"Processing Iceye_{index}...")

    # Clean
    cv2.imwrite(f"{save_dir}/1_clean.png", img_bgr)

    # Traditional
    count_t, mask = detect_ships_traditional(path)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_bgr, f"Count: {count_t}", (100, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 20)
    cv2.imwrite(f"{save_dir}/2_traditional.png", mask_bgr)
    del mask, mask_bgr

    # Pretrained
    count_p, res_p = detect_pretrained(path)
    vis_p = visualize_object_predictions(
        image=img_rgb.copy(), 
        object_prediction_list=res_p.object_prediction_list
    )["image"]
    cv2.imwrite(f"{save_dir}/3_pretrained.png", cv2.cvtColor(vis_p, cv2.COLOR_RGB2BGR))
    del res_p, vis_p

    # Expert
    count_e, res_e = detect_expert(path)
    vis_e = visualize_object_predictions(
        image=img_rgb.copy(), 
        object_prediction_list=res_e.object_prediction_list
    )["image"]
    cv2.imwrite(f"{save_dir}/4_expert.png", cv2.cvtColor(vis_e, cv2.COLOR_RGB2BGR))
    del res_e, vis_e, img_bgr, img_rgb

    print(f"Exported index {index}: Trad({count_t}) | Exp({count_e})")
    gc.collect()

def run_bulk_export(num_images=7):
    """
    Iterates through all images to generate the full portfolio gallery.
    """
    print(f"Starting Bulk Export for {num_images} images...")
    for i in range(num_images):
        export_panels_for_index(i)
    print("\nAll detection panels have been exported successfully.")

if __name__ == "__main__":
    run_bulk_export()
