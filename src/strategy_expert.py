from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np

def detect_expert(image_path, model_path='models/expert_weights.pt'):

    """Uses custom SAR weights with Sliced Aided Hyper Inference (SAHI)."""
    
    # Load model trained on OpenSARShip_2
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.25,
        device='cpu'
    )
    

    # Walk through the whole image in 640x640 chunks
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    
    ship_count = len(result.object_prediction_list)
    
    # Convert SAHI result to a viewable image
    return ship_count, result
