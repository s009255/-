import numpy as np
import cv2

img1 = cv2.imread("little.jpg", -1)
img2 = cv2.imread("tra_img.png", -1) # this one has transparency
h, w, depth = img2.shape

result = np.zeros((h, w, 3), np.uint8)

for i in range(h):
    for j in range(w):
        color1 = img1[i, j]
        color2 = img2[i, j]
        alpha = color2[3] / 255.0
        new_color = [ (1 - alpha) * color1[0] + alpha * color2[0],
                      (1 - alpha) * color1[1] + alpha * color2[1],
                      (1 - alpha) * color1[2] + alpha * color2[2] ]
        result[i, j] = new_color

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()