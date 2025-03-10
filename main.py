from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QPixmap, QImage
import sys
from PIL import Image
import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageQt
import traceback
from edgedetectors import EdgeDetector
from PyQt5.QtCore import QBuffer, QIODevice
import PIL.ImageQt as ImageQtModule


# Manually patch QBuffer and QIODevice into ImageQt
ImageQtModule.QBuffer = QBuffer
ImageQtModule.QIODevice = QIODevice

# Load the UI file
ui, _ = loadUiType("edgeBoundary_Ui.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)
        
        self.image = None
        self.value = None
        self.sigma_value = 0

        self.image1_original = None  # Store original image1
        self.image2_original = None  # Store original image2

        # Initializing Buttons 
        self.filterUpload_button.clicked.connect(lambda: self.uploadImage(1))
        self.filterDownload_button.clicked.connect(self.downloadImage)

        # Initializing Sliders
        self.kernel_slider.sliderReleased.connect(self.handleFilter)
        self.sigma_slider.sliderReleased.connect(self.handleFilter)
        self.mean_slider.sliderReleased.connect(self.handleFilter)

        # Allow scaling of image
        self.original_image.setScaledContents(True)  
        self.filtered_image.setScaledContents(True)

        # For sigma (0.1-10 is a good range)
        # For sigma (0.5-5.0 is a more practical range for most images)
        self.sigma_slider.setMinimum(5)    # 0.5 after division
        self.sigma_slider.setMaximum(50)   # 5.0 after division
        self.sigma_slider.setValue(10)     # 1.0 after division (good default)

        # For low threshold ratio (0.01-0.3 is more practical)
        self.kernel_slider.setMinimum(1)   # 0.01 after division
        self.kernel_slider.setMaximum(30)  # 0.3 after division
        self.kernel_slider.setValue(6)     # 0.06 after division (good default)

        # For high threshold ratio (0.1-0.6 is more practical)
        self.mean_slider.setMinimum(10)    # 0.1 after division
        self.mean_slider.setMaximum(60)    # 0.6 after division
        self.mean_slider.setValue(15)      # 0.15 after division (good default)

    def handle_kernelSlider(self):
        # Get sigma value (divide by 10 for finer control)
        slider_value = self.sigma_slider.value()
        self.sigma_value = slider_value / 10.0
        # Update label with descriptive text
        self.sigma_label.setText(f"{self.sigma_value:.1f}")
        
        # Get low threshold value (divide by 100 for percentage)
        slider_value = self.kernel_slider.value()
        self.kernel_value = slider_value / 100.0
        # Update label with descriptive text
        self.kernel_label.setText(f"{self.kernel_value:.2f}")
        
        # Get high threshold value (divide by 100 for percentage)
        slider_value = self.mean_slider.value()
        self.mean_value = slider_value / 100.0
        # Update label with descriptive text
        self.mean_label.setText(f"{self.mean_value:.2f}")
        
        # Log values for debugging
        print(f"Edge detection parameters - Sigma: {self.sigma_value}, " 
            f"Low Threshold: {self.kernel_value}, High Threshold: {self.mean_value}")
        
        return [self.kernel_value, self.sigma_value, self.mean_value]
        
    def uploadImage(self, value):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        
        if file_path:
           
            self.value = value

            match value:
                case 1:
                    q_image, self.image = self.process_and_store_grayscale(file_path)  # Get QImage & NumPy array
                    self.original_image.setPixmap(QPixmap.fromImage(q_image))
                    self.filtered_image.setPixmap(QPixmap.fromImage(q_image))

                case 2:
                    return


            # Set scaled contents for each QLabel only once
            self.original_image.setScaledContents(True)
            self.filtered_image.setScaledContents(True)


                            
        print("upload")

    def downloadImage(self):
        if self.value is None:
            print("No image uploaded yet. Please upload an image before downloading.")
            return
        
        # Mapping value to QLabel attributes
        image_mapping = {
            1: self.filtered_image,
            2: self.rgbGray_image,
            3: self.histogramOriginal_image,
            4: self.hyprid_image
        }
        
        label = image_mapping.get(self.value)

        if not label or label.pixmap() is None:
            print("No valid image found to download.")
            return
        
        pixmap = label.pixmap()

        # Open save dialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.bmp *.jpeg)", options=options)
        
        if file_path:
            if pixmap and not pixmap.isNull():
                pixmap.save(file_path)
                print(f"Image saved to {file_path}")
            else:
                print("No valid image found in QLabel.")
    
    def handleFilter(self):
        try:
            if self.image is None:
                self.kernel_slider.setValue(3)
                self.sigma_slider.setValue(1)
                self.mean_slider.setValue(1)
                raise ValueError("No image loaded. Please upload an image before applying edge detection.")

            # Get slider values
            self.sliderValues = self.handle_kernelSlider()
            
            # Apply Canny edge detection
            sigma = self.sigma_value
            low_threshold_ratio = self.kernel_value  # Use kernel slider for low threshold ratio
            high_threshold_ratio = self.mean_value  # Use mean slider for high threshold ratio
            
            # Modify the EdgeDetector class to accept threshold parameters
            q_image = EdgeDetector.apply_edge_detection(
                self.image, 
                'canny', 
                sigma=sigma,
                low_thresh_ratio=low_threshold_ratio,
                high_thresh_ratio=high_threshold_ratio
            )
            
            # Update the filtered image
            self.filtered_image.setPixmap(QPixmap.fromImage(q_image))

        except ValueError as ve:
            print(f"Error: {ve}")

    def process_and_store_grayscale(self, file_path):
       
        original_image = Image.open(file_path).convert("RGB")
        img_array = np.array(original_image)

        # Convert to grayscale using standard formula
        grayscale_values = (
            0.299 * img_array[:, :, 0] +
            0.587 * img_array[:, :, 1] +
            0.114 * img_array[:, :, 2]
        )
        grayscale_array = grayscale_values.astype(np.uint8)

        # Convert NumPy grayscale array to QImage
        height, width = grayscale_array.shape
        bytes_per_line = width
        q_image = QImage(grayscale_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        return q_image, grayscale_array 



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
