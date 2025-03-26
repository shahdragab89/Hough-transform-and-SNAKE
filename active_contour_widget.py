from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage
from scipy.ndimage import gaussian_gradient_magnitude
import numpy as np

def gaussian_kernel(size, sigma=1.0):
    
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

class ActiveContourWidget:
    def __init__(self, ui):
        self.ui = ui
        self.image = None
        self.contour = None
        self.radius = 100
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 1.0
        self.iterations = 100
        self.num_points = 100
        self.edge_map = None
        self.chain_code = ""

        self.ui.radius_spinBox.setMaximum(500)  
        self.ui.alpha_slider.setMaximum(200)   
        self.ui.beta_slider.setMaximum(200)     
        self.ui.gamma_slider.setMaximum(300)    
        self.ui.iteration_slider.setMaximum(1000)  
        self.ui.contour_points_slider.setMaximum(700)  
        
        # Connect UI elements
        self.ui.snakeApply_button.clicked.connect(self.apply_active_contour)
        
        self.ui.radius_spinBox.valueChanged.connect(self.update_radius)
        self.ui.alpha_slider.valueChanged.connect(self.update_alpha)
        self.ui.beta_slider.valueChanged.connect(self.update_beta)
        self.ui.gamma_slider.valueChanged.connect(self.update_gamma)
        self.ui.iteration_slider.valueChanged.connect(self.update_iterations)
        self.ui.contour_points_slider.valueChanged.connect(self.update_num_points)
        
        # Initialize UI labels with default values
        self.ui.radius_spinBox.setValue(self.radius)
        self.ui.alpha_slider.setValue(int(self.alpha * 10))
        self.ui.beta_slider.setValue(int(self.beta * 10))
        self.ui.gamma_slider.setValue(int(self.gamma * 10))
        self.ui.iteration_slider.setValue(self.iterations)
        self.ui.contour_points_slider.setValue(self.num_points)
        self.ui.alpha_label.setText(f"{self.alpha:.1f}")
        self.ui.beta_label.setText(f"{self.beta:.1f}")
        self.ui.gamma_label.setText(f"{self.gamma:.1f}")
        self.ui.iteration_label.setText(f"{self.iterations}")
        self.ui.contour_points_label.setText(f"{self.num_points}")
    
    def set_image(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3: 
            # if rgb then apply grayscale conversion formula
            self.image = (0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]).astype(np.uint8)
        else:
            self.image = image  

        self.compute_edge_map()
        self.draw_initial_contour()
        self.ui.resultImage_snake.clear()
        self.ui.area_label.setText("0")
        self.ui.perimeter_label.setText("0")  
    
    # def compute_edge_map(self):
    #     blurred = gaussian_gradient_magnitude(self.image.astype(float), sigma=1.5)
    #     self.edge_map = blurred / blurred.max()

    def convolve(self, image, kernel):
        kernel_size = kernel.shape[0]
        pad_size = kernel_size // 2
        padded_image = np.pad(image, pad_size, mode='reflect')
        output = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.sum(region * kernel)
        
        return output

    def compute_edge_map(self):
        if self.image is None:
            return
        
        sigma = 1.5  
        kernel_size = int(6 * sigma) | 1  

        gaussian_filter = gaussian_kernel(kernel_size, sigma)
        smoothed = self.convolve(self.image.astype(float), gaussian_filter)

        gx = np.zeros_like(smoothed)
        gy = np.zeros_like(smoothed)

        gx[:, :-1] = smoothed[:, 1:] - smoothed[:, :-1]  
        gy[:-1, :] = smoothed[1:, :] - smoothed[:-1, :] 

        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        self.edge_map = gradient_magnitude / gradient_magnitude.max()

    def draw_initial_contour(self):
        if self.image is None:
            return
        
        height, width = self.image.shape
        center = (width // 2, height // 2)
        theta = np.linspace(0, 2 * np.pi, self.num_points)
        x = center[0] + self.radius * np.cos(theta)
        y = center[1] + self.radius * np.sin(theta)
        self.contour = np.vstack((x, y)).T.astype(int)
        
        self.display_image_with_contour()
    
    def display_image_with_contour(self):
        if self.image is None or self.contour is None:
            return
        
        image_copy = np.stack((self.image,)*3, axis=-1)  
        
        for i, pt in enumerate(self.contour):
            x, y = pt
            if 0 <= x < image_copy.shape[1] and 0 <= y < image_copy.shape[0]:
                image_copy[y-1:y+2, x-1:x+2] = [0, 0, 255]  
                
        
        q_image = self.numpy_to_qimage(image_copy)
        self.ui.inputImage_snake.setPixmap(QPixmap.fromImage(q_image))
    
    def numpy_to_qimage(self, array):
        height, width, channel = array.shape
        bytes_per_line = channel * width
        return QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def reset_contour(self):
        self.draw_initial_contour()
        self.ui.resultImage_snake.clear()
        self.ui.area_label.setText("0")
        self.ui.perimeter_label.setText("0")
        
    def apply_active_contour(self):
        if self.image is None or self.contour is None:
            return

        self.reset_contour()

        for _ in range(self.iterations):
            self.contour = self.update_contour(self.contour)
        
        self.chain_code = self.compute_chain_code(self.contour)
        self.display_result_image()
        self.calculate_area_perimeter()
    
    def update_contour(self, contour):
        new_contour = np.copy(contour)
        N = len(contour)  

        for i in range(N):
            min_energy = float('inf')
            best_point = contour[i]

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_point = contour[i] + np.array([dx, dy])

                    elasticity_energy = self.alpha * ((new_point[0] - contour[(i-1) % N][0])**2 + 
                                                      (new_point[1] - contour[(i-1) % N][1])**2)

                    smoothness_energy = self.beta * (((contour[(i+1) % N][0] - 2 * new_point[0] + contour[(i-1) % N][0])**2) + 
                                                     ((contour[(i+1) % N][1] - 2 * new_point[1] + contour[(i-1) % N][1])**2))

                    external_energy = -self.gamma * self.get_pixel_intensity(new_point)

                    total_energy = elasticity_energy + smoothness_energy + external_energy 

                    if total_energy < min_energy:
                        min_energy = total_energy
                        best_point = new_point

            new_contour[i] = best_point
        return new_contour

    def compute_chain_code(self, contour):
        directions = {(0, 1): '0', (1, 1): '1', (1, 0): '2', (1, -1): '3', (0, -1): '4', (-1, -1): '5', (-1, 0): '6', (-1, 1): '7'}
        chain_code = ""
        for i in range(len(contour) - 1):
            dx = contour[i+1][0] - contour[i][0]
            dy = contour[i+1][1] - contour[i][1]
            direction = directions.get((np.sign(dx), np.sign(dy)), "")
            chain_code += direction
            
        print(chain_code) 
        return chain_code


    def get_pixel_intensity(self, point):
        x, y = point
        if 0 <= x < self.edge_map.shape[1] and 0 <= y < self.edge_map.shape[0]:
            return self.edge_map[y, x]  
        return 0
    
    def display_result_image(self):
        if self.image is None or self.contour is None:
            return
        
        result_image = np.stack((self.image,)*3, axis=-1)
        for pt in self.contour:
            x, y = pt
            if 0 <= x < result_image.shape[1] and 0 <= y < result_image.shape[0]:
                result_image[y-1:y+2, x-1:x+2] = [255, 0, 0]  
        
        q_image = self.numpy_to_qimage(result_image)
        self.ui.resultImage_snake.setPixmap(QPixmap.fromImage(q_image))
        self.ui.resultImage_snake.setScaledContents(True)


    def calculate_area_perimeter(self):
        x = self.contour[:, 0]
        y = self.contour[:, 1]
        
        # Compute perimeter using Euclidean distance
        perimeter = np.sum(np.sqrt(np.diff(x, append=x[0])**2 + np.diff(y, append=y[0])**2))
        
        # Compute area using Shoelace formula
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        self.ui.area_label.setText(f"{area:.1f}")
        self.ui.perimeter_label.setText(f"{perimeter:.1f}")

    def update_radius(self, value):
        self.radius = value
        self.ui.radius_spinBox.setValue(self.radius)
        self.draw_initial_contour()
    
    def update_alpha(self, value):
        self.alpha = value / 10.0
        self.ui.alpha_label.setText(f"{self.alpha:.1f}")
    
    def update_beta(self, value):
        self.beta = value / 10.0
        self.ui.beta_label.setText(f"{self.beta:.1f}")
    
    def update_gamma(self, value):
        self.gamma = value / 10.0
        self.ui.gamma_label.setText(f"{self.gamma:.1f}")
    
    def update_iterations(self, value):
        self.iterations = value
        self.ui.iteration_label.setText(f"{self.iterations}")
    
    def update_num_points(self, value):
        self.num_points = value
        self.ui.contour_points_label.setText(f"{self.num_points}")
        self.draw_initial_contour()
