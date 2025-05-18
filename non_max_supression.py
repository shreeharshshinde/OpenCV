import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('sceptile.png', cv2.IMREAD_GRAYSCALE)

# Step 1: Noise Reduction using Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# Step 2: Compute Gradients using Sobel Operator
sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction

# Compute magnitude and angle of gradients
magnitude = np.sqrt(sobelx**2 + sobely**2)
angle = np.arctan2(sobely, sobelx) * 180 / np.pi

def non_maximum_suppression(magnitude, angle):
    M, N = magnitude.shape
    output = np.zeros((M, N), dtype=np.uint8)  # Initialize output image
    angle = angle % 180  # Make sure angles are within [0, 180] degrees

    for i in range(1, M-1):  # Ignore the borders
        for j in range(1, N-1):
            # Check the gradient direction and suppress non-maximum values
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbors = [magnitude[i, j+1], magnitude[i, j-1]]  # Horizontal neighbors
            elif 22.5 <= angle[i, j] < 67.5:
                neighbors = [magnitude[i+1, j-1], magnitude[i-1, j+1]]  # Diagonal neighbors (bottom-left to top-right)
            elif 67.5 <= angle[i, j] < 112.5:
                neighbors = [magnitude[i+1, j], magnitude[i-1, j]]  # Vertical neighbors
            else:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]  # Diagonal neighbors (top-left to bottom-right)

            # Suppress the current pixel if it's not larger than both neighbors
            if magnitude[i, j] >= max(neighbors):
                output[i, j] = magnitude[i, j]
            else:
                output[i, j] = 0

    return output

# Apply Non-Maximum Suppression
nms_image = non_maximum_suppression(magnitude, angle)

# Visualize the result of Non-Maximum Suppression
plt.imshow(nms_image, cmap='gray')
plt.title('Non-Maximum Suppression')
plt.axis('off')
plt.show()
