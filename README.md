# SAR Ship Detection: ICEYE Open Dataset Benchmark

I set up this project to evaluate the performance of different detection methods on the ICEYE open spotlight dataset. The goal was to test if specialized SAR training is a functional necessity or if traditional computer vision and off-the-shelf models can handle high-resolution radar data.

This repository implements a processing pipeline for raw ICEYE data and compares three specific detection strategies against 100MP scenes of the Panama Canal entrance.

## Dataset and Sensor Specifications
The test set consists of 7 images acquired by the ICEYE-X50 constellation. These scenes cover the Pacific Ocean entrance to the Panama Canal, featuring high vessel density and varying environmental conditions.

- **Platform:** ICEYE-X50 (Spotlight Mode)
- **Acquisition Period:** November 2025
- **Sensor Specs:** X-band (9.65 GHz) | VV Polarization
- **Geometry:** Right-looking descending orbits. The dataset includes variable incidence angles (~25° to 30°) and off-nadir angles to test model robustness across different viewing perspectives.
- **Resolution:** 10,000 × 10,000 pixels (100MP per scene)

## Technical Pipeline

### Linear Percentile Normalization
Raw 16-bit ICEYE TIFFs have a massive dynamic range that makes standard inference difficult. I found that logarithmic scaling—while better for human viewing—compressed the ship signatures too much for the models. 

I used a **[2, 98] percentile clipping** strategy to normalize the data. This stretches the pixel intensities based on the actual distribution of each specific scene, isolating metallic glint from sea clutter without the non-linear distortion of a log curve.

### Sliced Aided Hyper Inference (SAHI)
Because vessels are extremely small relative to the 10,000-pixel frame, standard downscaling would destroy the target features. I integrated **SAHI** to run inference on overlapping 640x640 windows. This allows the models to process the area at full resolution.

## Detection Strategies Benchmarked
I compared three different approaches to see where the capabilities break down:

1. **Traditional Morphology:** Using Scikit-Image to isolate high-intensity blobs. It is effective at finding bright returns but lacks semantic context, leading to false positives from wave crests and speckle noise.
2. **Pretrained COCO YOLOv8:** Used as a baseline control. Despite being a standard for optical images, it fails to detect ships in this dataset because SAR backscatter signatures do not match its learned features for boats.
3. **Expert SAR-YOLO:** A model trained specifically on the **OpenSARShip_2** dataset. It is optimized to recognize the radiometric signatures of vessels in X-band imagery.

## Repository Structure
- `data/`: 7 SAR images (TIFF and PNG) depicting the Panama Canal entrance.
- `src/`: Core logic for data preparation, normalization, and the results gallery viewer.
- `models/`: Weights for the OpenSARShip_2 model and the baseline pretrained YOLOv8n.
- `output/`: Comparative results for all 3 approaches across the 7-image test set.

## Usage
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Prepare data:** `python src/prepare_data.py`  
   *(Downloads raw 16-bit TIFFs and applies percentile normalization)*
3. **Run benchmark:** `python scripts/export_results.py`
4. **View results:** `python src/view_gallery.py --index 0`
