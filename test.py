    # -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:42:47 2018

@author: TH
"""

# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import argparse
import utils
import cv2
from PIL import Image
'''
def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    plt.figure()
    plt.axis("off")
    plt.imshow(image_4channel)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255
    plt.figure()
    plt.axis("off")
    plt.imshow(white_background_image)

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    plt.figure()
    plt.axis("off")
    plt.imshow(final_image)
    return final_image
    #return final_image.astype(np.uint8)
'''

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
args = vars(ap.parse_args())
 
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)
#r,g,b,a = cv2.split(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
r,g,b = cv2.split(image)
#image = cv2.merge([r,g,b,a])

plt.figure()
plt.axis("off")
plt.imshow(image)

# show our image
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(image2)

meanV = cv2.mean(V)

print(meanV)

image = image.reshape((image.shape[0] * image.shape[1], 3))

clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)
 
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()  