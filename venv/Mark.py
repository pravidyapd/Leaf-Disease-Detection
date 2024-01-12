import cv2
import numpy as np

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform adaptive thresholding to obtain binary image
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded

def detect_leaf_disease(filename):
    # Load the image
    image = cv2.imread(filename)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Perform Canny edge detection
    edges = cv2.Canny(processed_image, 30, 100)

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black image to draw the contour
    contour_image = np.zeros_like(image)

    # Iterate through the contours and draw the boundary of the cancer region
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Draw the contours with green color

    # Blend the contour image with the original image
    blended_image = cv2.addWeighted(image, 0.7, contour_image, 0.3, 0)

    # Display the blended image
    cv2.imshow("Disease Region", blended_image)

# Example usage:
