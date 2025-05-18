# An image grdaient is a directional change in the intensity or color in an image.
import cv2
import numpy as np
from matplotlib import pyplot as plt

img =cv2.imread("sceptile.png", cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
# we are using a 64bit float due the negative slope induced by transforming the image from white to black. Supports thenegativ enumber that we will be dealing with.  
lap = np.uint8(np.absolute(lap))

sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
edges = cv2.Canny(img, 100, 200)
# prewitt edge detection 
kernelX = np.array([[1,1,1],[0,0,0],[-1,-1,1]])
kernelY = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
PrewittX = cv2.filter2D(img, 0, kernelX)
PrewittY = cv2.filter2D(img, 0, kernelY)
PrewittX = np.uint8(np.absolute(PrewittX))
PrewittY = np.uint8(np.absolute(PrewittY))




sobelCombined = cv2.bitwise_or(sobelX, sobelY)

titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined', 'Canny','PrewittX','PrewittY']
images = [img, lap, sobelX, sobelY, sobelCombined, edges, PrewittX, PrewittY]
for i in range(8):
    plt.subplot(2,4,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()