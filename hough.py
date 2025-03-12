import cv2
import numpy as np
from edgedetectors import EdgeDetector

class Hough:
    @staticmethod
    def detect_lines(image, low_threshold, high_threshold, votes):
        """
        General Edge Detection: low_threshold = 50, high_threshold = 150
        Fine Details (Weak Edges): low_threshold = 30, high_threshold = 100
        Only Strongest Edges: low_threshold = 100, high_threshold = 200
        """
        if image is None:
            print("Error: Image is None.")
            return None
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()  

        # Apply Canny edge detection (merge with baty!!)  ------>
        img = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(img, low_threshold, high_threshold)

        # Apply Hough Line Transform
        lines = Hough.hough_lines(edges, 1, np.pi / 180, votes)
        output_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) 

        # Draw the detected lines on the image
        if lines is None:
            print("No lines detected.")
            return output_image

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return output_image
    

    @staticmethod
    def hough_lines(edges, rho_resolution=1, theta_resolution=np.pi/180, threshold=100):
        """
        Custom implementation of the Hough Transform for line detection.
        :param edges: Binary edge image (from Canny edge detection)
        :param rho_resolution: Resolution of the rho axis (default 1 pixel)
        :param theta_resolution: Resolution of the theta axis (default 1 degree in radians)
        :param threshold: Minimum votes required to consider a line
        :return: List of (rho, theta) values of detected lines
        """
        height, width = edges.shape
        diag_len = int(np.sqrt(height**2 + width**2)) 
        rhos = np.arange(-diag_len, diag_len, rho_resolution)
        thetas = np.arange(-np.pi / 2, np.pi / 2, theta_resolution)
        
        # Accumulator (rho, theta space)
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
        y_idxs, x_idxs = np.nonzero(edges) 
        
        # Vote in the accumulator
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            for theta_idx, theta in enumerate(thetas):
                rho = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len  
                accumulator[rho, theta_idx] += 1
        
        # Extract lines that pass the threshold
        line_indices = np.argwhere(accumulator >= threshold)
        lines = [(rhos[rho_idx], thetas[theta_idx]) for rho_idx, theta_idx in line_indices]
        
        return np.array(lines).reshape(-1, 1, 2)  

