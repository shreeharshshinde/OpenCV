import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('sceptile.png', cv2.IMREAD_GRAYSCALE)

# Step 1: Noise Reduction using Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# Step 2: Compute Gradient (Sobel Operator)
sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude and Angle of Gradients
magnitude = np.sqrt(sobelx**2 + sobely**2)
angle = np.arctan2(sobely, sobelx) * 180 / np.pi

# Step 3: Non-Maximum Suppression (not explicitly implemented here)

# Step 4: Double Thresholding
# Define high and low thresholds
low_threshold = 50
high_threshold = 150

# Apply Double Thresholding
strong_edges = (magnitude > high_threshold).astype(np.uint8)
weak_edges = ((magnitude >= low_threshold) & (magnitude <= high_threshold)).astype(np.uint8)

# Visualize the steps
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(strong_edges, cmap='gray')
plt.title('Strong Edges (High Threshold)')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(weak_edges, cmap='gray')
plt.title('Weak Edges (Low Threshold)')

plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
