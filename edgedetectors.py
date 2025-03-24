import numpy as np
from PyQt5.QtGui import QImage
from scipy.ndimage import gaussian_filter, binary_dilation

class EdgeDetector:
    result = None  
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
            for x in range(image_w): #loop pixel by pixel in padded image
                local_region = padded_image[y:y+kernel_h, x:x+kernel_w] #Extract a part from the padded image equal to the kernel size
                result[y, x] = np.sum(local_region * kernel) #Convolution Operation
        return result

    @staticmethod
    def apply_edge_detection(image, method, sigma=1, low_thresh_ratio=0.05, high_thresh_ratio=0.15):
        if image is None:
            raise ValueError("No image provided for edge detection")
        
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected an image as a NumPy array")
        
        img = image.copy()
        
        #normalize to 0-255 if needed
        if img.max() > 0:  #to avoid division by zero
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
        result = None
        
        if method.lower() == 'canny':
            #in case of RGB image, process each channel separately 
            if len(img.shape) == 3 and img.shape[2] >= 3:
                #initialize result with same shape as input but with only one channel
                result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                
                #apply Canny to channels separately
                for i in range(3):  #process RGB channels
                    channel_result = EdgeDetector.canny(img[:,:,i], sigma, low_thresh_ratio, high_thresh_ratio)
                    #combining channel results one by one using for loop
                    result = np.maximum(result, channel_result) #compares the initial result array with each channel canny result and take maximum
            else:
                result = EdgeDetector.canny(img, sigma, low_thresh_ratio, high_thresh_ratio)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        height, width = result.shape
        bytes_per_line = width
        EdgeDetector.result = result
        return QImage(result.data, width, height, bytes_per_line, QImage.Format_Grayscale8)


    @staticmethod
    def canny(image, sigma, low_thresh_ratio=0.05, high_thresh_ratio=0.15):
        #1st step: gaussian blur to reduce noise
        smoothed = gaussian_filter(image, sigma=sigma)
        
        #2nd step: grad mag & direction using sobel filters
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) #changes in horizontal direction -> vertical edges
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) #changes in vertical direction -> horizontal edges

        grad_x = EdgeDetector.apply_kernel(smoothed, sobel_x)  
        grad_y = EdgeDetector.apply_kernel(smoothed, sobel_y)

        magnitude = np.hypot(grad_x, grad_y)
        magnitude = (magnitude / magnitude.max()) * 255 
        direction = np.arctan2(grad_y, grad_x) 
        
        #3rd step: non-max suppression
        angle = np.rad2deg(direction) % 180  #convert to degrees and normalize to 0-180
        
        # create directional masks - Initialize nms (non maximum suppression) output by zeros
        nms = np.zeros_like(magnitude, dtype=np.float32)
        
        #quantize angles to 4 directions: 0, 45, 90, 135 degrees
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
        # pad magnitude array to handle edge pixels
        magnitude_padded = np.pad(magnitude, pad_width=1, mode='constant', constant_values=0) #pad 1 pixel at each size of constant = 0
        
        # copy magnitude to nms initially
        nms = magnitude.copy()
        
        # create masks for each direction and apply them in a vectorized way
        # 1. horizontal direction (0 degrees)
        mask_h = (angle_quantized == 0)
        # compare to left & right pixels
        left = magnitude_padded[1:-1, :-2][mask_h] # column to the left of center pixels
        right = magnitude_padded[1:-1, 2:][mask_h] # column to the right of center pixels
        center_h = magnitude[mask_h] #centered zero pixels only
        # suppress non-maximum pixels
        nms[mask_h] = np.where((center_h >= left) & (center_h >= right), center_h, 0) #if center is brighter than left and right, keep it, else, suppress to 0
        
        # 2. diagonal direction (45 degrees)
        mask_d1 = (angle_quantized == 1)
        # compare to diagonal pixels
        top_left = magnitude_padded[:-2, :-2][mask_d1]
        bottom_right = magnitude_padded[2:, 2:][mask_d1]
        center_d1 = magnitude[mask_d1]
        # suppress non-maximum pixels
        nms[mask_d1] = np.where((center_d1 >= top_left) & (center_d1 >= bottom_right), center_d1, 0)
        
        # 3. vertical direction (90 degrees)
        mask_v = (angle_quantized == 2)
        # compare to top & bottom pixels
        top = magnitude_padded[:-2, 1:-1][mask_v]
        bottom = magnitude_padded[2:, 1:-1][mask_v]
        center_v = magnitude[mask_v]
        # suppress non-maximum pixels
        nms[mask_v] = np.where((center_v >= top) & (center_v >= bottom), center_v, 0)
        
        # 4.other diagonal direction (135 degrees)
        mask_d2 = (angle_quantized == 3)
        # compare to other diagonal pixels
        top_right = magnitude_padded[:-2, 2:][mask_d2]
        bottom_left = magnitude_padded[2:, :-2][mask_d2]
        center_d2 = magnitude[mask_d2]
        #suppress non-maximum pixels
        nms[mask_d2] = np.where((center_d2 >= top_right) & (center_d2 >= bottom_left), center_d2, 0)
        
        #4th step: hysteresis thresholding with vectorized operations
        max_val = np.max(nms) # get the max value in the nms result
        high_thresh = max_val * high_thresh_ratio #T_high ratio from max value
        low_thresh = max_val * low_thresh_ratio #T_low ratio from max value
        
        # create binary masks for strong and weak edges
        strong_edges = (nms >= high_thresh).astype(np.uint8) #strong edges are pixels above the high threshold
        weak_edges = ((nms >= low_thresh) & (nms < high_thresh)).astype(np.uint8) #weak edges are pixels between low and high thresholds
        #(np.uint8) converts the boolean masks to 8-bit unsigned integers (0 or 1)

        # use binary dilation to connect weak edges to strong edges efficiently instead of pixel-by-pixel iteration
        # create a 3x3 structural element for dilation
        structure = np.ones((3, 3), dtype=np.uint8)
        
        # iteratively dilate strong edges and find intersection with weak edges
        final_edges = strong_edges.copy()
        prev_final = np.zeros_like(final_edges)
        
        # continue dilating until no more weak edges are connected to strong edges
        # or a maximum number of iterations is reached (to prevent infinite loops)
        max_iterations = 10
        iteration = 0
        
        #iterating to connect weak&strong edges
        while not np.array_equal(prev_final, final_edges) and iteration < max_iterations:
            iteration += 1
            #store prev iteration result
            prev_final = final_edges.copy()
            #dilate current strong edges
            dilated = binary_dilation(final_edges, structure=structure)
            #connect weak edges in dilated region
            final_edges = dilated & (weak_edges | strong_edges)
        
        #convert to 0-255 range for display
        final_edges = final_edges.astype(np.uint8) * 255
        
        return final_edges