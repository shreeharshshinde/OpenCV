# canny edge sdetector is an edge detector oeprator that uses a multi-stage algprithm
#  to detect a wide range of edges in images.
# simple code that loads an image using matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("messi5.png", 0)
canny = cv2.Canny(img, 100, 200)
# threshold values should be provided of the hysterisis procedure
titles= ['image', 'canny']
images = [img, canny]
for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()