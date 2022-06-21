import cv2
import numpy as np 
filename = "image.jpeg"
img = cv2.imread(filename) # dosyayi oku
from matplotlib import pyplot as plt
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale kopya
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # rgb kopya
plt.imshow(im, cmap='gray')
plt.show()


white = cv2.inRange(im,200,255)
plt.imshow(white, cmap='gray')
plt.show()

# Gaussian Blur

blurred = cv2.GaussianBlur(white,(5,5),0.8)
plt.imshow(blurred, cmap='gray')
plt.show()

edge_image = cv2.Canny(blurred,50,150)
plt.imshow(edge_image, cmap='gray')
plt.show()


# ROI 
mask = np.zeros_like(edge_image)
vertices = np.array([[(150,525),(440,320),(520,330),(920,525)]],np.int32)
print (vertices)
cv2.fillPoly(mask, vertices, 255)

plt.imshow(mask, cmap='gray')
plt.show()

print (edge_image.shape, mask.shape)
masked = cv2.bitwise_and(edge_image, mask)
plt.imshow(masked, cmap='gray')
plt.show()


lines = cv2.HoughLinesP(masked,2,np.pi/180,20,np.array([]),minLineLength=50,maxLineGap=200)
zeros = np.zeros_like(img)
print (lines)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(zeros,(x1,y1),(x2,y2),(0,0,255),4)

img = cv2.addWeighted(img,0.8,zeros, 1.0,0.)
plt.imshow(img)
plt.show()
