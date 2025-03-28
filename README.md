# Edge and Boundary Detection (Hough Transform and SNAKE)**   

## **Overview**  
This assignment focuses on detecting edges and boundaries in grayscale and color images using the **Canny edge detector, Hough Transform**, and **Active Contour Model (SNAKE)**. The goal is to detect and highlight geometric shapes, as well as extract object contours for further analysis.  

## **Tasks to Implement**  

### **1. Edge Detection and Shape Detection**  
For all given images:  
- Apply the **Canny Edge Detector** to extract edges.  
- Detect and highlight **lines, circles, and ellipses** using the **Hough Transform** (if present in the images).  
- Superimpose the detected shapes onto the original images for visualization.  

### **2. Active Contour Model (SNAKE) Implementation**  
For given images:  
- **Initialize a contour** around a specific object in the image.  
- Use the **Greedy Algorithm** to evolve the **Active Contour Model (Snake)** to refine the contour.  
- Represent the final contour using **Chain Code**.  
- Compute and report:  
  - **Perimeter** of the detected contour.  
  - **Area** enclosed within the contour.  

## **Setup & Dependencies**  
Ensure the following Python libraries are installed:  
```bash
pip install numpy opencv-python matplotlib scikit-image scipy
```
Or, if using Jupyter Notebook, include:  
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters, feature
from scipy.ndimage import measurements
```

## **How to Run the Code**  
1. Load an input image (grayscale or color).  
2. Run the scripts for edge detection, shape detection, and active contour model.  
3. Save and analyze the results.  

