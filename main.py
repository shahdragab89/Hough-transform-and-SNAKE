from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QPixmap, QImage
import sys
from PIL import Image
import numpy as np
from PIL import Image
from edgedetectors import EdgeDetector
from PyQt5.QtCore import QBuffer, QIODevice
import PIL.ImageQt as ImageQtModule
from active_contour_widget import ActiveContourWidget 
from hough import Hough
from PyQt5.QtGui import QImage, QPixmap
from worker import Worker

ImageQtModule.QBuffer = QBuffer
ImageQtModule.QIODevice = QIODevice

ui, _ = loadUiType("edgeBoundary_Ui.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)
        
        self.image = None
        self.value = None
        self.sigma_value = 0

        self.image1_original = None  # store original image1
        self.image2_original = None  # store original image2

        self.cannyUpload_button.clicked.connect(lambda: self.uploadImage(1))
        self.cannyDownload_button.clicked.connect(self.downloadImage)
        self.houghUpload_button.clicked.connect(lambda: self.uploadImage(2))
        self.snakeUpload_button.clicked.connect(lambda: self.uploadImage(3)) 
        self.houghClear_button.clicked.connect(self.clear_hough)
        self.houghApply_button.clicked.connect(self.handleHough)

        self.lines_radioButton.clicked.connect(lambda: self.handleRadio("lines"))
        self.circles_radioButton.clicked.connect(lambda: self.handleRadio("circles"))
        self.ellipses_radioButton.clicked.connect(lambda: self.handleRadio("ellipses"))

        self.low_threshold_slider.sliderReleased.connect(self.handleCanny)
        self.sigma_slider.sliderReleased.connect(self.handleCanny)
        self.high_threshold_slider.sliderReleased.connect(self.handleCanny)
        self.frame_48.hide()
        
        self.original_image.setScaledContents(True)  
        self.canny_result_image.setScaledContents(True)

        self.sigma_slider.setMinimum(5)    # 0.5 after division
        self.sigma_slider.setMaximum(50)   # 5.0 after division
        self.sigma_slider.setValue(10)     # 1.0 after division (good default)

        self.low_threshold_slider.setMinimum(1)   # 0.01 after division
        self.low_threshold_slider.setMaximum(30)  # 0.3 after division
        self.low_threshold_slider.setValue(6)     # 0.06 after division (good default)

        self.high_threshold_slider.setMinimum(10)    # 0.1 after division
        self.high_threshold_slider.setMaximum(60)    # 0.6 after division
        self.high_threshold_slider.setValue(15)      # 0.15 after division (good default)

        self.active_contour_widget = ActiveContourWidget(self)
        self.worker = Worker(self)

    def handle_sliders_values(self):
        #get sigma value (divide by 10 for finer control)
        slider_value = self.sigma_slider.value()
        self.sigma_value = slider_value / 10.0
        #update labels with vals
        self.sigma_label.setText(f"{self.sigma_value:.1f}")
        
        #get t_low value (divide by 100 for %)
        slider_value = self.low_threshold_slider.value()
        self.low_threshold_value = slider_value / 100.0
        #update labels with vals
        self.low_threshold_label.setText(f"{self.low_threshold_value:.2f}")
        
        #get t_high value (divide by 100 for %)
        slider_value = self.high_threshold_slider.value()
        self.high_threshold_value = slider_value / 100.0
        #update labels with vals
        self.high_threshold_label.setText(f"{self.high_threshold_value:.2f}")
        
        #log values for test
        print(f"Edge detection parameters - Sigma: {self.sigma_value}, " 
            f"Low Threshold: {self.low_threshold_value}, High Threshold: {self.high_threshold_value}")
        
        return [self.low_threshold_value, self.sigma_value, self.high_threshold_value]
        
    def uploadImage(self, value):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.webp);;All Files (*)", options=options)
        
        if file_path:
           
            self.value = value

            match value:
                case 1:
                    q_image, self.image = self.process_and_store_image(file_path)  
                    self.original_image.setPixmap(QPixmap.fromImage(q_image))
                    try:
                        edge_image = EdgeDetector.apply_edge_detection(
                            self.image,  
                            'canny', 
                            sigma=1.0,  #default sigma
                            low_thresh_ratio=0.05,  #default T_low
                            high_thresh_ratio=0.15  #default T_high
                        )
                        self.canny_result_image.setPixmap(QPixmap.fromImage(edge_image))
                    except Exception as e:
                        print(f"Error applying edge detection: {e}")
                        self.canny_result_image.setPixmap(QPixmap.fromImage(q_image))

                case 2:
                    self.q_image, self.image = self.process_and_store_image(file_path)  
                    self.inputImage_hough.setPixmap(QPixmap.fromImage(self.q_image))
                    self.resultImage_hough.setPixmap(QPixmap.fromImage(self.q_image))
                    self.clear_hough()
                case 3:
                    q_image, self.image = self.process_and_store_image(file_path)  
                    self.inputImage_snake.setPixmap(QPixmap.fromImage(q_image))
                    self.active_contour_widget.set_image(self.image)

            #set scaled contents for each QLabel only once
            self.original_image.setScaledContents(True)
            self.canny_result_image.setScaledContents(True)
            self.inputImage_hough.setScaledContents(True)
            self.resultImage_hough.setScaledContents(True)
            self.inputImage_snake.setScaledContents(True)

        print("upload")

    def downloadImage(self):
        if self.value is None:
            print("No uploaded img, upload first!")
            return
        
        #map the value to QLabel attributes
        image_mapping = {
            1: self.canny_result_image,
            2: self.rgbGray_image,
            3: self.histogramOriginal_image,
            4: self.hyprid_image
        }
        
        label = image_mapping.get(self.value)

        if not label or label.pixmap() is None:
            print("No img found")
            return
        
        pixmap = label.pixmap()

        #open save dialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.bmp *.jpeg)", options=options)
        
        if file_path:
            if pixmap and not pixmap.isNull():
                pixmap.save(file_path)
                print(f"Image saved to {file_path}")
            else:
                print("no QImage found")
    
    def handleCanny(self):
        try:
            if self.image is None:
                self.low_threshold_slider.setValue(3)
                self.sigma_slider.setValue(1)
                self.high_threshold_slider.setValue(1)
                raise ValueError("No uploaded img, upload first!")

            self.sliderValues = self.handle_sliders_values()
            
            #take parameter values from sliders
            sigma = self.sigma_value
            low_threshold_ratio = self.low_threshold_value  
            high_threshold_ratio = self.high_threshold_value 
            
            #apply Canny edge detection
            q_image = EdgeDetector.apply_edge_detection(
                self.image, 
                'canny', 
                sigma=sigma,
                low_thresh_ratio=low_threshold_ratio,
                high_thresh_ratio=high_threshold_ratio
            )
            #update canny result image
            self.canny_result_image.setPixmap(QPixmap.fromImage(q_image))

        except ValueError as ve:
            print(f"Error: {ve}")

    def process_and_store_image(self, file_path):
        original_image = Image.open(file_path).convert("RGB")
        img_array = np.array(original_image)
        
        #convert PIL image to QImage
        height, width, channels = img_array.shape
        bytes_per_line = channels * width
        q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        return q_image, img_array

    def convert_numpy_to_qimage(self, numpy_image):
        height, width = numpy_image.shape[:2]
        
        if len(numpy_image.shape) == 2:  #grayscale
            bytes_per_line = width
            return QImage(numpy_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  #color (BGR)
            bytes_per_line = 3 * width
            return QImage(numpy_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

    def handleRadio(self, label):
        self.label = label
        self.worker.update_label(self.label)

    def handleHough(self):
        if self.label == "lines":
            if self.image is None:
                print("Error: Could not read the image.")
                return
        
            self.worker.update_label("lines")
            lowThreshold, highThreshold, votes,_ = self.worker.get_slider_values()
            result_image = Hough.detect_lines(self.image, lowThreshold, highThreshold, votes)

        elif self.label == "circles":
            self.worker.update_label("circles")
            minRadius, maxRadius, _, _ = self.worker.get_slider_values()
            result_image = Hough.hough_circles(self.image, min_radius=minRadius, max_radius=maxRadius)
            
        elif self.label == "ellipses":
            self.worker.update_label("ellipses")
            lowThreshold, highThreshold, minAxis, maxAxis = self.worker.get_slider_values()
            result_image = Hough.hough_ellipses(self.image, low_threshold=lowThreshold, high_threshold=highThreshold, min_axis=minAxis, max_axis=maxAxis)

        #convert to QImage then set to QPixmap
        qimage = self.convert_numpy_to_qimage(result_image)
        self.resultImage_hough.setPixmap(QPixmap.fromImage(qimage))

    def clear_hough(self):
        self.worker.clear()
        self.inputImage_hough.setPixmap(QPixmap.fromImage(self.q_image))
        self.resultImage_hough.setPixmap(QPixmap.fromImage(self.q_image))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())