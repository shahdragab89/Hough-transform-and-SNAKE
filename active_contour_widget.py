from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage
import numpy as np

class ActiveContourWidget:
    def __init__(self, ui):
        self.ui = ui
        self.image = None
        self.contour = None
        self.radius = 100
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 1.0
        self.balloon = 0.5
        self.iterations = 100
        self.num_points = 100
        self.edge_map = None

        # Update maximum values for UI controls
        self.ui.radius_spinBox.setMaximum(500)  # Increased max from 99 to 500
        self.ui.alpha_slider.setMaximum(200)    # Allows finer alpha tuning (0.0 - 10.0)
        self.ui.beta_slider.setMaximum(200)     # Allows finer beta tuning (0.0 - 10.0)
        self.ui.gamma_slider.setMaximum(200)    # Allows finer gamma tuning (0.0 - 10.0)
        self.ui.iteration_slider.setMaximum(500)  # More iterations allowed
        self.ui.contour_points_slider.setMaximum(500)  # More contour points
        
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
        self.image = image
        self.compute_edge_map()
        self.draw_initial_contour()

    def compute_edge_map(self):
        """ Compute gradient magnitude as an edge strength map. """
        gx, gy = np.gradient(self.image.astype(float))
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        self.edge_map = gradient_magnitude / gradient_magnitude.max()  # Normalize to [0,1]
    
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
        
        image_copy = np.stack((self.image,)*3, axis=-1)  # Convert grayscale to RGB manually
        
        # Draw the contour as a blue circle (thicker)
        for i, pt in enumerate(self.contour):
            x, y = pt
            if 0 <= x < image_copy.shape[1] and 0 <= y < image_copy.shape[0]:
                image_copy[y-1:y+2, x-1:x+2] = [0, 0, 255]  # Blue, thicker
                
        
        q_image = self.numpy_to_qimage(image_copy)
        self.ui.inputImage_snake.setPixmap(QPixmap.fromImage(q_image))
    
    def numpy_to_qimage(self, array):
        height, width, channel = array.shape
        bytes_per_line = channel * width
        return QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    def apply_active_contour(self):
        if self.image is None or self.contour is None:
            return
        
        for _ in range(self.iterations):
            self.update_internal_energy_weights()
            self.contour = self.update_contour(self.contour)
            self.display_result_image()
    
    def update_contour(self, contour):
        """ Updates the contour by minimizing energy. """
        new_contour = np.copy(contour)
        N = len(contour)  # Number of contour points

        for i in range(N):
            min_energy = float('inf')
            best_point = contour[i]

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_point = contour[i] + np.array([dx, dy])

                    # Elasticity Energy
                    elasticity_energy = self.alpha * ((new_point[0] - contour[(i-1) % N][0])**2 + 
                                                      (new_point[1] - contour[(i-1) % N][1])**2)

                    # Smoothness Energy
                    smoothness_energy = self.beta * (((contour[(i+1) % N][0] - 2 * new_point[0] + contour[(i-1) % N][0])**2) + 
                                                     ((contour[(i+1) % N][1] - 2 * new_point[1] + contour[(i-1) % N][1])**2))

                    # External Energy
                    external_energy = -self.gamma * self.get_pixel_intensity(new_point)

                    # Balloon Force (expansion or contraction)
                    balloon_force = self.balloon * (1 if self.balloon > 0 else -1)

                    # Total Energy
                    total_energy = elasticity_energy + smoothness_energy + external_energy + balloon_force

                    if total_energy < min_energy:
                        min_energy = total_energy
                        best_point = new_point

            new_contour[i] = best_point
        return new_contour

    def update_internal_energy_weights(self):
        """ Dynamically adjusts elasticity and smoothness based on edge proximity. """
        avg_edge_strength = np.mean([self.get_pixel_intensity(pt) for pt in self.contour])

        if avg_edge_strength < 0.2:
            self.alpha = min(self.alpha + 0.1, 2.0)
            self.beta = max(self.beta - 0.1, 0.5)
        elif avg_edge_strength > 0.5:
            self.alpha = max(self.alpha - 0.1, 0.5)
            self.beta = min(self.beta + 0.1, 2.0)

        self.ui.alpha_slider.setValue(int(self.alpha * 10))
        self.ui.beta_slider.setValue(int(self.beta * 10))
        self.ui.alpha_label.setText(f"{self.alpha:.1f}")
        self.ui.beta_label.setText(f"{self.beta:.1f}")


    def get_pixel_intensity(self, point):
        """ Returns edge strength from the computed edge map. """
        x, y = point
        if 0 <= x < self.edge_map.shape[1] and 0 <= y < self.edge_map.shape[0]:
            return self.edge_map[y, x]  # Use edge strength
        return 0
    
    def display_result_image(self):
        if self.image is None or self.contour is None:
            return
        
        result_image = np.stack((self.image,)*3, axis=-1)
        for pt in self.contour:
            x, y = pt
            if 0 <= x < result_image.shape[1] and 0 <= y < result_image.shape[0]:
                result_image[y-1:y+2, x-1:x+2] = [255, 0, 0]  # Red for final contour
        
        q_image = self.numpy_to_qimage(result_image)
        self.ui.resultImage_snake.setPixmap(QPixmap.fromImage(q_image))
        self.ui.resultImage_snake.setScaledContents(True)

    
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
