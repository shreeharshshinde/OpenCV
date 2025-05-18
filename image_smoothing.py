# Gaussian filter is nothing but using different-weight-kernel, in both x and y direction.
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('sceptile.png', 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



gblur = cv2.GaussianBlur(img, (5,5), 0)

titles = ['image',  'GaussianBlur']
images = [img,gblur]

for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()