import cv2
import numpy as np
from edgedetectors import EdgeDetector
from filters import FilterProcessor
from scipy.spatial import ConvexHull

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
        
        colored_image = image.copy()
        correct_color = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()  

        img = FilterProcessor.gaussian_filter(gray, 5, 1.5)
        edges = cv2.Canny(img, low_threshold, high_threshold)

        # Apply Hough Line Transform
        lines = Hough.hough_lines(edges, 1, np.pi / 180, votes)

        if lines is None:
            print("No lines detected.")
            return correct_color

        # To draw the detected lines on the image
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
            correct_color = Hough.draw_line(correct_color, x1, y1, x2, y2, (0, 255, 0), 10)

        return correct_color
    

    @staticmethod
    def hough_lines(edges, rho_resolution=1, theta_resolution=np.pi/180, threshold=100):
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
        
        # FInd the lines that are bigger than the threshold
        line_indices = np.argwhere(accumulator >= threshold)
        lines = [(rhos[rho_idx], thetas[theta_idx]) for rho_idx, theta_idx in line_indices]
        
        return np.array(lines).reshape(-1, 1, 2)  
    
    def draw_line(image, x1, y1, x2, y2, color, thickness=1):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy  

        while True:
            # Draw a pixel at (x1, y1)
            if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]:  
                image[y1, x1] = color  

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return image


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
        # Convert to grayscale
        if len(img.shape) == 3 and img.shape[2] == 3:
            # img = Hough.enhance_contrast(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur and Canny Edge Detection
        # img = cv2.GaussianBlur(img, (5, 5), 1.5)
        img = FilterProcessor.gaussian_filter(img, 5, 1.5)
        # img = EdgeDetector.apply_edge_detection(img, method='canny', low_thresh_ratio=50/255, high_thresh_ratio=150/255)
        img = cv2.Canny(img, 50, 150)

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
        Draw detected circles on the original image, ensuring the output is in RGB format.

        :param A: Accumulator array containing detected circles
        :param img: Original image
        :return: Image with detected circles drawn in RGB format
        """
        if img is None:
            print("Error: Image is None.")
            return None

        # Copy the original image and convert to RGB
        colored_image = img.copy()
        correct_color = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

        circleCoordinates = np.argwhere(A)  # Extract circle information
        for r, x, y in circleCoordinates:
            cv2.circle(correct_color, (y, x), r, color=(0, 255, 0), thickness=2)

        return correct_color  # Return the properly formatted image


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
            threshold=13,  # Increased from 8 to 15 to reduce false positives
            region=5,     # Decreased from 15 to 10 for better accuracy
            radius=[max_radius, min_radius]
        )

        return Hough.displayCircles(circles, src)



    @staticmethod
    def fit_ellipse_manual(points, max_iter=100, tolerance=1e-6):
        """
        Fit an ellipse to a set of points using an iterative least squares approach.
        :param points: Contour points (Nx2 numpy array)
        :param max_iter: Maximum iterations for convergence
        :param tolerance: Convergence threshold
        :return: (center_x, center_y), (major_axis, minor_axis), angle
        """
        if len(points) < 5:
            return None  # Not enough points to fit an ellipse

        # Compute centroid
        centroid = np.mean(points, axis=0)
        x_c, y_c = centroid

        # Shift points to centroid
        shifted_points = points - centroid

        # Compute covariance matrix
        cov_matrix = np.cov(shifted_points.T)

        # Eigen decomposition (for axes and orientation)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Ensure positive eigenvalues
        eigenvalues = np.abs(eigenvalues)

        # Identify major and minor axes
        major_index = np.argmax(eigenvalues)
        minor_index = 1 - major_index

        # --- FIX: Scale the axes properly ---
        scaling_factor = 2.5  # Adjust this to fit the outer boundary better
        major_axis_length = scaling_factor * np.sqrt(eigenvalues[major_index])
        minor_axis_length = scaling_factor * np.sqrt(eigenvalues[minor_index])

        # Ensure major > minor
        if major_axis_length < minor_axis_length:
            major_axis_length, minor_axis_length = minor_axis_length, major_axis_length

        # Get ellipse orientation
        major_eigenvector = eigenvectors[:, major_index]
        angle = np.degrees(np.arctan2(major_eigenvector[1], major_eigenvector[0]))

        # --- FIX: Use convex hull to refine ellipse fitting ---
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]  # Get outer boundary points

        # Compute max distances from centroid to outer points for better approximation
        distances = np.linalg.norm(hull_points - centroid, axis=1)
        max_distance = np.max(distances)

        # Adjust ellipse size based on max distance
        major_axis_length = max(major_axis_length, 2 * max_distance)
        minor_axis_length = max(minor_axis_length, 2 * max_distance * 0.6)  # Keep proportion

        return (x_c, y_c), (major_axis_length, minor_axis_length), angle



    @staticmethod
    def hough_ellipses(image, low_threshold, high_threshold, min_axis, max_axis):
        """
        Detect ellipses in a colored image while keeping the result in color (RGB format).

        :param image: Input color image (RGB)
        :param low_threshold: Lower threshold for edge detection
        :param high_threshold: Upper threshold for edge detection
        :param min_axis: Minimum axis length for valid ellipses
        :param max_axis: Maximum axis length for valid ellipses
        :return: Image with detected ellipses drawn in RGB format
        """
        if image is None:
            print("Error: Image is None.")
            return None

        if len(image.shape) == 2:
            print("Error: Input image is grayscale, expected a colored image.")
            return None

        # Copy the original image and convert to RGB
        colored_image = image.copy()
        correct_color = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        blurred = FilterProcessor.gaussian_filter(gray, 5, 1.5)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        # edges = EdgeDetector.apply_edge_detection(blurred, method='canny', low_thresh_ratio=low_threshold/255, high_thresh_ratio=high_threshold/255)
        print(edges)
        print(type(edges))


        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) >= 5:  # Ellipse fitting requires at least 5 points
                contour_points = contour[:, 0, :]  # Extract x, y coordinates

                ellipse = Hough.fit_ellipse_manual(contour_points)
                if ellipse:
                    (x, y), (major_axis, minor_axis), angle = ellipse

                    # Check ellipse constraints
                    if min_axis <= major_axis <= max_axis and min_axis <= minor_axis <= max_axis:
                        cv2.ellipse(correct_color, ((x, y), (major_axis, minor_axis), angle), (0, 255, 0), 2)  # Draw ellipse

        return correct_color  # Return the correctly formatted image
