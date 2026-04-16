import numpy as np
import cv2
from skimage import measure, morphology

def detect_ships_traditional(image_path):

    img = cv2.imread(image_path, 0).astype(float)
    s_norm = img / 255.0  # Bring back to 0-1 range

    mask = s_norm > 0.9

    # remove_small_objects is the 'killer feature' here to kill wave clutter
    mask = morphology.remove_small_objects(mask, min_size=1000)
    mask = morphology.binary_closing(mask)

    # Label and Count
    labels = measure.label(mask)
    properties = measure.regionprops(labels)
    
    ship_count = len(properties)


    return ship_count, (mask * 255).astype(np.uint8)
