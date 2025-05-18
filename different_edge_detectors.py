import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('sceptile.jpg', cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)

# Laplacian edge Detector
lap = np.uint8(np.absolute(lap))
# Sobel Edge Detection (X and Y)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y
sobelx = np.uint8(np.absolute(sobelx))
sobely = np.uint8(np.absolute(sobely))
sobel_combined = cv2.bitwise_or(sobelx, sobely)  # Combined Sobel

# Prewitt Edge Detection (X and Y)
prewitt_kernel_y = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewittx = cv2.filter2D(image, -1, prewitt_kernel_x)  # Prewitt X
prewitty = cv2.filter2D(image, -1, prewitt_kernel_y)  # Prewitt Y
prewittx = np.uint8(np.absolute(prewittx))
prewitty = np.uint8(np.absolute(prewitty))
prewitt_combined = cv2.bitwise_or(prewittx, prewitty)  # Combined Prewitt

# Roberts Edge Detection
roberts_kernel_x = np.array([[1, 0], [0, -1]])
roberts_kernel_y = np.array([[0, 1], [-1, 0]])
robertsx = cv2.filter2D(image, -1, roberts_kernel_x)  # Roberts X
robertsy = cv2.filter2D(image, -1, roberts_kernel_y)  # Roberts Y
robertsx = np.uint8(np.absolute(robertsx))
robertsy = np.uint8(np.absolute(robertsy))
roberts_combined = cv2.bitwise_or(robertsx, robertsy) 


titles = ['Original Image', 'Sobel X', 'Sobel Y', 'Combined Sobel',
          'Prewitt X', 'Prewitt Y', 'Combined Prewitt',
          'Roberts X', 'Roberts Y', 'Combined Roberts', 'Laplacian']

images = [image, sobelx, sobely, sobel_combined,
          prewittx, prewitty, prewitt_combined,
          robertsx, robertsy, roberts_combined, lap]

plt.figure(figsize=(14, 10))
for i in range(len(images)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
