import random
import statistics as stat
import cv2
import numpy as np
import scipy.signal as sig
from PyQt5.QtGui import QPixmap, QImage
from scipy.ndimage import convolve 
from PyQt5.QtCore import QBuffer, QIODevice
import PIL.ImageQt as ImageQtModule

# Manually patch QBuffer and QIODevice into ImageQt
ImageQtModule.QBuffer = QBuffer
ImageQtModule.QIODevice = QIODevice



class FilterProcessor:
    @staticmethod
    def average_filter(image, kernel_size = 3):
        try:
            # Make sure kernel_size is odd
            if kernel_size % 2 == 0:
                raise ValueError("mask_size must be an odd number.")
            # Make the kernel a tuple
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            # Create kernel
            kernel = np.ones(kernel_size, dtype=np.float32) / (kernel_size[0] * kernel_size[1])

            # Padding the kernel with edges value 
            filtered_image = convolve(image, kernel, mode='nearest')

            return filtered_image.astype(np.uint8)  

        except ValueError as ve:
            print(f" Error: {ve}")
        
    @staticmethod
    def gaussian_filter(image, kernel_size=3, sigma=1):
        try:
            # Make sure mask_size is odd
            if kernel_size % 2 == 0:
                raise ValueError("mask_size must be an odd number.")

            # Create Gaussian kernel
            k = kernel_size // 2  # Kernel radius
            x, y = np.mgrid[-k:k+1, -k:k+1]

            # Applying Gaussian function to the kernel and normalizing the kernel
            gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))  
            gaussian_kernel /= np.sum(gaussian_kernel) 

            # Padding the Gaussian kernel with edge value and applying convolution
            filtered_image = convolve(image, gaussian_kernel, mode='nearest')
            
            return filtered_image.astype(np.uint8)
        
        except ValueError as ve:
            print(f" Error: {ve}")

        
    @staticmethod
    def median_filter(image, kernel_size=3):
        try:
            # Make sure kernel_size is odd
            if kernel_size % 2 == 0:
                raise ValueError("filter_size must be an odd number.")

            # Pad the image with edges value
            pad_size = kernel_size // 2
            padded_image = np.pad(image, pad_size, mode='edge')
            filtered_image = np.zeros_like(image, dtype=np.uint8)
            rows, cols = image.shape

            # Apply median filter
            for i in range(rows):
                for j in range(cols):
                    # Take the specified kernel size from the image 
                    region = padded_image[i:i+kernel_size, j:j+kernel_size]
                    # Exchage the pixel value with the median value 
                    filtered_image[i, j] = np.median(region)

            return filtered_image
        
        except ValueError as ve:
            print(f" Error: {ve}")

        
    @staticmethod
    def applyFilterAndDisplay(image, filterType, sliderValues):
        print("apply filter")
        if filterType == "Average": 
            filtered_image = FilterProcessor.average_filter(image, sliderValues[0])
        elif filterType == "Gaussian":
            filtered_image = FilterProcessor.gaussian_filter(image, sliderValues[0], sliderValues[1])
        elif filterType == "Median": 
            filtered_image = FilterProcessor.median_filter(image, sliderValues[0])
        else:
            raise ValueError("Invalid filterType. Choose 'Average', 'Gaussian', or 'Median'.")

        # Convert to QImage
        height, width = filtered_image.shape
        bytes_per_line = width
        filtered_image = QImage(filtered_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        return filtered_image
        