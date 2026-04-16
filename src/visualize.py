import cv2
import matplotlib.pyplot as plt

def plot_sar(index, title="SAR View"):

    image_path = f'data/test_set/iceye_{index}.png'
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find {image_path}")
        return
    
    # Convert BGR (OpenCV default) to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(img_rgb)
    plt.title(f"{title} {index} - Full Scene")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    print("Visualization module updated: Single Full Scene mode.")
