import cv2
import matplotlib.pyplot as plt
import os
import argparse

def show_results_for(index=0):
    """Displays the 4 exported panels for a specific ICEYE image."""
    path = f'output/detections/iceye_{index}'
    
    titles = [
        "1. Clean (Percentile Norm)", 
        "2. Traditional (Morphology)", 
        "3. Pretrained (COCO YOLO)", 
        "4. Expert (SAR-YOLO)"
    ]
    filenames = ["1_clean.png", "2_traditional.png", "3_pretrained.png", "4_expert.png"]
    
    # Check if directory exists
    if not os.path.exists(path):
        print(f"Error: Directory {path} not found. Ensure you have run the export script first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(4):
        full_path = os.path.join(path, filenames[i])
        if os.path.exists(full_path):
            img = cv2.imread(full_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(titles[i], fontsize=16, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, f"File not found:\n{filenames[i]}", 
                         ha='center', va='center', fontsize=14)
        
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAR ship detection results for a specific image index.")
    parser.add_argument("--index", type=int, default=0, help="The index of the ICEYE image to visualize (0-6).")
    
    args = parser.parse_args()
    show_results_for(args.index)
