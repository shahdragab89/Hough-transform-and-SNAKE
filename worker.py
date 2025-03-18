from hough import Hough

class Worker:
    def __init__(self, main_window):
        self.main_window = main_window  # Store reference to MainWindow
        self.label = None
        self.main_window.hough_slider1.valueChanged.connect(self.get_slider_values)
        self.main_window.hough_slider2.valueChanged.connect(self.get_slider_values)
        self.main_window.hough_slider3.valueChanged.connect(self.get_slider_values)
        self.main_window.hough_slider4.valueChanged.connect(self.get_slider_values)

    def update_label(self, label):
        self.label = label
        if label == "lines":
            self.main_window.hough_sliderLabel1.setText("Low Threshold")
            self.main_window.hough_sliderLabel2.setText("High Threshold")
            self.main_window.hough_sliderLabel3.setText("Votes")
            self.main_window.hough_sliderLabel3.show()
            self.main_window.frame_47.show()
            self.main_window.frame_48.hide()
            self.main_window.hough_sliderLabel4.hide()

            self.update_slider_ranges()

        elif label == "circles":
            self.main_window.hough_sliderLabel1.setText("Minimun Radius")
            self.main_window.hough_sliderLabel2.setText("Maximum Radius")
            # self.main_window.hough_sliderLabel3.setText("DP")
            self.main_window.hough_sliderLabel3.hide()
            self.main_window.hough_sliderLabel4.hide()
            self.main_window.frame_47.hide()
            self.main_window.frame_48.hide()
            self.update_slider_ranges()

        elif label == "ellipses":
            self.main_window.hough_sliderLabel1.setText("Low Threshold")
            self.main_window.hough_sliderLabel2.setText("High Threshold")
            self.main_window.hough_sliderLabel3.setText("Min Axis")
            self.main_window.hough_sliderLabel4.setText("Max Axis")
            self.main_window.hough_sliderLabel3.show()
            self.main_window.hough_sliderLabel4.show()
            self.main_window.frame_47.show()
            self.main_window.frame_48.show()

    def get_slider_values(self):
        value1 = self.main_window.hough_slider1.value()
        value2 = self.main_window.hough_slider2.value()
        value3 = self.main_window.hough_slider3.value()
        value4 = self.main_window.hough_slider4.value()

        self.main_window.hough_sliderValue1.setText(str(value1))
        self.main_window.hough_sliderValue2.setText(str(value2))
        self.main_window.hough_sliderValue3.setText(str(value3))
        self.main_window.hough_sliderValue4.setText(str(value4))

        # result_image = Hough.detect_lines(self.main_window.image, value1, value2, value3)
        

        # return value1, value2, value3
        return value1, value2, value3, value4
    
    def update_slider_ranges(self):
        if self.label == "lines":
            self.main_window.hough_slider1.setRange(10, 150)  # Set range
            self.main_window.hough_slider2.setRange(50, 255)
            self.main_window.hough_slider3.setRange(1, 300)
        elif self.label == "circles":
            self.main_window.hough_slider1.setRange(1, 100)  # Min Radius
            self.main_window.hough_slider2.setRange(10, 200)  # Max Radius
        elif self.label == "ellipses":
            self.main_window.hough_slider1.setRange(0, 255)  # Set range
            self.main_window.hough_slider2.setRange(0, 255)
            self.main_window.hough_slider3.setRange(10, 100)
            self.main_window.hough_slider4.setRange(50, 300)

    def clear(self):
        self.main_window.hough_slider1.setValue(0)  
        self.main_window.hough_slider2.setValue(0)
        self.main_window.hough_slider3.setValue(0)
        self.main_window.hough_slider4.setValue(0)