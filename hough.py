import cv2
import numpy as np
from edgedetectors import EdgeDetector
from filters import FilterProcessor

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

        img = FilterProcessor.gaussian_filter(gray, 5, 1.5)
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
    

    @staticmethod
    def enhance_contrast(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced_lab,cv2.COLOR_LAB2BGR)


    @staticmethod
    def detectCircles(img, threshold, region, radius=None):
        """
        Detect circles in an image using a Hough Transform approach.

        :param img: Input image
        :param threshold: Minimum votes required to consider a valid circle
        :param region: Region around a detected circle to find maxima
        :param radius: Range of radius values as [max_radius, min_radius]
        :return: Accumulator array indicating detected circles
        """
        # Convert to grayscale if necessary
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = Hough.enhance_contrast(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur and Canny Edge Detection
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        img = cv2.Canny(img, 100, 200)

        # Image dimensions
        (M, N) = img.shape

        # Define radius range
        if radius is None:
            R_max = np.max((M, N))
            R_min = 3
        else:
            [R_max, R_min] = radius

        R = R_max - R_min

        # Initialize accumulator arrays with padding to avoid overflow issues
        A = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))
        B = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))

        # Precompute angles for circle perimeter
        theta = np.deg2rad(np.arange(0, 360))
        edges = np.argwhere(img)  # Get edge pixel coordinates

        for val in range(R):
            r = R_min + val
            # Create a circle blueprint
            bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
            (m, n) = (r + 1, r + 1)  # Center of the blueprint

            for angle in theta:
                x = int(np.round(r * np.cos(angle)))
                y = int(np.round(r * np.sin(angle)))
                bprint[m + x, n + y] = 1

            constant = np.count_nonzero(bprint)

            # Update accumulator array
            for x, y in edges:
                X = [x - m + R_max, x + m + R_max]  # Compute extreme X values
                Y = [y - n + R_max, y + n + R_max]  # Compute extreme Y values
                A[r, X[0]:X[1], Y[0]:Y[1]] += bprint

            A[r][A[r] < threshold * constant / max(r, 1)] = 0  # Prevent division by zero

        # Find local maxima in the accumulator
        for r, x, y in np.argwhere(A):
            temp = A[r - region:r + region, x - region:x + region, y - region:y + region]
            try:
                p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
            except:
                continue
            B[r + (p - region), x + (a - region), y + (b - region)] = 1

        return B[:, R_max:-R_max, R_max:-R_max]

    @staticmethod
    def displayCircles(A, img):
        """
        Draw detected circles on the original image.

        :param A: Accumulator array containing detected circles
        :param img: Original image
        :return: Image with detected circles drawn
        """
        circleCoordinates = np.argwhere(A)  # Extract circle information
        for r, x, y in circleCoordinates:
            cv2.circle(img, (y, x), r, color=(0, 255, 0), thickness=2)
        return img


    @staticmethod
    def hough_circles(source: np.ndarray, min_radius: int = 20, max_radius: int = 50) -> np.ndarray:
        """
        Apply the Hough Circle Detection on the given image with optimized parameters.

        :param source: Input image
        :param min_radius: Minimum circle radius
        :param max_radius: Maximum circle radius
        :return: Image with detected circles drawn
        """
        src = np.copy(source)

        # Apply optimized detection
        circles = Hough.detectCircles(
            src, 
            threshold=15,  # Increased from 8 to 15 to reduce false positives
            region=10,     # Decreased from 15 to 10 for better accuracy
            radius=[max_radius, min_radius]
        )

        return Hough.displayCircles(circles, src)
