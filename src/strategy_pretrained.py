from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2

def detect_pretrained(image_path):

    """Standard COCO YOLOv8 with Slicing (SAHI)."""
    
    # Load standard weights (trained on everyday objects)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='models/yolov8n.pt',
        confidence_threshold=0.25,
        device='cpu' 
    )
    
    # Slice the big image into 640x640 chunks
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    
    # COCO Class 8 is 'boat'. 
    # We filter to see if it thinks SAR signatures are boats.
    boat_count = 0
    for pred in result.object_prediction_list:
        if pred.category.id == 8:
            boat_count += 1
            
    return boat_count, result
