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

# Step 3: Double Thresholding (Strong and Weak edges)
low_threshold = 50
high_threshold = 150

strong_edges = (magnitude > high_threshold).astype(np.uint8)
weak_edges = ((magnitude >= low_threshold) & (magnitude <= high_threshold)).astype(np.uint8)

# Step 4: Non-Maximum Suppression (code from previous step)
def non_maximum_suppression(magnitude, angle):
    M, N = magnitude.shape
    output = np.zeros((M, N), dtype=np.uint8)
    angle = angle % 180  # Make sure angles are within [0, 180] degrees

    for i in range(1, M-1):  # Ignore borders
        for j in range(1, N-1):
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbors = [magnitude[i, j+1], magnitude[i, j-1]]  # Horizontal neighbors
            elif 22.5 <= angle[i, j] < 67.5:
                neighbors = [magnitude[i+1, j-1], magnitude[i-1, j+1]]  # Diagonal neighbors (bottom-left to top-right)
            elif 67.5 <= angle[i, j] < 112.5:
                neighbors = [magnitude[i+1, j], magnitude[i-1, j]]  # Vertical neighbors
            else:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]  # Diagonal neighbors (top-left to bottom-right)

            if magnitude[i, j] >= max(neighbors):
                output[i, j] = magnitude[i, j]
            else:
                output[i, j] = 0

    return output

nms_image = non_maximum_suppression(magnitude, angle)

# Step 5: Hysteresis
def hysteresis(strong_edges, weak_edges):
    M, N = strong_edges.shape
    final_edges = np.copy(strong_edges)

    for i in range(1, M-1):
        for j in range(1, N-1):
            if weak_edges[i, j]:
                # If there is a strong edge in the neighborhood, convert weak edge to strong
                if (strong_edges[i+1, j-1] or strong_edges[i+1, j] or strong_edges[i+1, j+1]
                    or strong_edges[i, j-1] or strong_edges[i, j+1]
                    or strong_edges[i-1, j-1] or strong_edges[i-1, j] or strong_edges[i-1, j+1]):
                    final_edges[i, j] = 1
                else:
                    final_edges[i, j] = 0
    return final_edges

# Apply Hysteresis
final_edges = hysteresis(strong_edges, weak_edges)

# Visualize the final edge-detected image
plt.imshow(final_edges, cmap='gray')
plt.title('Final Edges (Hysteresis)')
plt.axis('off')
plt.show()
