import numpy as np
from PyQt5.QtGui import QImage
from scipy.ndimage import gaussian_filter, maximum_filter, binary_dilation

class EdgeDetector:
    @staticmethod
    def apply_kernel(image, kernel):
        #get dimensions of image, kernel 
        image_h, image_w = image.shape
        kernel_h, kernel_w = kernel.shape
        
        #padding calc
        pad_h = kernel_h // 2
        pad_w = kernel_w // 2
        
        #applying padding
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        #initialize result by zero array
        result = np.zeros_like(image, dtype=np.float32)
        
        #CONVOLUTION!!!
        for y in range(image_h):
            for x in range(image_w):
                local_region = padded_image[y:y+kernel_h, x:x+kernel_w]
                result[y, x] = np.sum(local_region * kernel)        
        return result

    @staticmethod
    def apply_edge_detection(image, method, sigma=1, low_thresh_ratio=0.05, high_thresh_ratio=0.15):
        if image is None:
            raise ValueError("No image provided for edge detection")
        
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected an image as a NumPy array")
        
        # Convert img to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            img = image.copy()
        
        # Normalize to 0-255 
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
        result = None
        
        if method.lower() == 'canny':
            result = EdgeDetector.canny(img, sigma, low_thresh_ratio, high_thresh_ratio)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        height, width = result.shape
        bytes_per_line = width
        return QImage(result.data, width, height, bytes_per_line, QImage.Format_Grayscale8)


    @staticmethod
    def canny(image, sigma, low_thresh_ratio=0.05, high_thresh_ratio=0.15):
        # 1st step: gaussian blur
        smoothed = gaussian_filter(image, sigma=sigma)
        
        # 2nd step: grad mag & direction
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        grad_x = EdgeDetector.apply_kernel(smoothed, sobel_x)
        grad_y = EdgeDetector.apply_kernel(smoothed, sobel_y)

        magnitude = np.hypot(grad_x, grad_y)
        magnitude = (magnitude / magnitude.max()) * 255 
        direction = np.arctan2(grad_y, grad_x) 
        
        # 3rd step: IMPROVED non-max suppression
        angle = np.rad2deg(direction) % 180  # Convert to degrees and normalize to 0-180
        
        # Create directional masks - Initialize nms output
        nms = np.zeros_like(magnitude, dtype=np.float32)
        rows, cols = magnitude.shape
        
        # Quantize the angles to four directions: 0, 45, 90, 135 degrees
        angle_quantized = np.zeros_like(angle, dtype=np.uint8)
        # 0 degrees (horizontal)
        angle_quantized[(angle >= 0) & (angle < 22.5) | (angle >= 157.5) & (angle < 180)] = 0
        # 45 degrees (diagonal)
        angle_quantized[(angle >= 22.5) & (angle < 67.5)] = 1
        # 90 degrees (vertical)
        angle_quantized[(angle >= 67.5) & (angle < 112.5)] = 2
        # 135 degrees (diagonal)
        angle_quantized[(angle >= 112.5) & (angle < 157.5)] = 3
        
        # Vectorized non-maximum suppression based on angle_quantized
        # Pad the magnitude array to handle edge pixels
        magnitude_padded = np.pad(magnitude, pad_width=1, mode='constant', constant_values=0)
        
        # Copy magnitude to nms initially
        nms = magnitude.copy()
        
        # Create masks for each direction and apply them in a vectorized way
        # Horizontal direction (0 degrees)
        mask_h = (angle_quantized == 0)
        # Compare with left and right pixels
        left = magnitude_padded[1:-1, :-2][mask_h]
        right = magnitude_padded[1:-1, 2:][mask_h]
        center_h = magnitude[mask_h]
        # Suppress non-maximum pixels
        nms[mask_h] = np.where((center_h >= left) & (center_h >= right), center_h, 0)
        
        # Diagonal direction (45 degrees)
        mask_d1 = (angle_quantized == 1)
        # Compare with diagonal pixels
        top_left = magnitude_padded[:-2, :-2][mask_d1]
        bottom_right = magnitude_padded[2:, 2:][mask_d1]
        center_d1 = magnitude[mask_d1]
        # Suppress non-maximum pixels
        nms[mask_d1] = np.where((center_d1 >= top_left) & (center_d1 >= bottom_right), center_d1, 0)
        
        # Vertical direction (90 degrees)
        mask_v = (angle_quantized == 2)
        # Compare with top and bottom pixels
        top = magnitude_padded[:-2, 1:-1][mask_v]
        bottom = magnitude_padded[2:, 1:-1][mask_v]
        center_v = magnitude[mask_v]
        # Suppress non-maximum pixels
        nms[mask_v] = np.where((center_v >= top) & (center_v >= bottom), center_v, 0)
        
        # Diagonal direction (135 degrees)
        mask_d2 = (angle_quantized == 3)
        # Compare with diagonal pixels
        top_right = magnitude_padded[:-2, 2:][mask_d2]
        bottom_left = magnitude_padded[2:, :-2][mask_d2]
        center_d2 = magnitude[mask_d2]
        # Suppress non-maximum pixels
        nms[mask_d2] = np.where((center_d2 >= top_right) & (center_d2 >= bottom_left), center_d2, 0)
        
        # 4th step: IMPROVED hysteresis thresholding with vectorized operations
        max_val = np.max(nms)
        high_thresh = max_val * high_thresh_ratio
        low_thresh = max_val * low_thresh_ratio
        
        # Create binary masks for strong and weak edges
        strong_edges = (nms >= high_thresh).astype(np.uint8)
        weak_edges = ((nms >= low_thresh) & (nms < high_thresh)).astype(np.uint8)
        
        # Use binary dilation to connect weak edges to strong edges efficiently
        # This replaces the pixel-by-pixel iteration
        # Create a 3x3 structural element for dilation
        structure = np.ones((3, 3), dtype=np.uint8)
        
        # Iteratively dilate strong edges and find intersection with weak edges
        final_edges = strong_edges.copy()
        prev_final = np.zeros_like(final_edges)
        
        # Continue dilating until no more weak edges are connected to strong edges
        # or a maximum number of iterations is reached (to prevent infinite loops)
        max_iterations = 10
        iteration = 0
        
        while not np.array_equal(prev_final, final_edges) and iteration < max_iterations:
            iteration += 1
            prev_final = final_edges.copy()
            # Dilate current strong edges
            dilated = binary_dilation(final_edges, structure=structure)
            # Connect any weak edges that are in the dilated region
            final_edges = dilated & (weak_edges | strong_edges)
        
        # Convert to 0-255 range for display
        final_edges = final_edges.astype(np.uint8) * 255
        
        return final_edges