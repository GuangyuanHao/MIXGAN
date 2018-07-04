from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import copy
import os
from scipy.io import loadmat as load
import numpy as np
import scipy
from PIL import Image
import cv2

def load_data(array):
    n =array.shape[0]
    # print("n",n)
    size = array[0].shape
    # print(size)
    imgA = array[0].reshape(1,size[0],size[1],size[2])
    # print(imgB.shape)
    for i in range(n-1):
        # print(array[i + 1][0].reshape(1, size[0], size[1], size[2]).shape)
        imgA = np.concatenate((imgA,array[i+1].reshape(1,size[0],size[1],size[2])),axis=0)
        # print(("[%d]"%(i)),imgB.shape)
    return imgA/127.5-1.0
def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299, 0.587, 0.144])
# imgA = load_data(q).astype(np.float32).reshape(32,32,3)
# imgA = rgb2gray(imgA)


# print(cannyA)
# print(np.max(imgA[1]))

def load_label(array):
    n =array.shape[0]
    hot_code = np.zeros(10).reshape(1,10)
    hot_code[0][array[0][1]]=1
    labelA = hot_code
    # print(hot_code)
    for i in range(n-1):
        hot_code = np.zeros(10).reshape(1, 10)
        hot_code[0][array[i+1][1]] = 1
        labelA = np.concatenate((labelA,hot_code),axis=0)
        # print(("[%d]"%(i)),labelA.shape)
    return labelA
# labelA= load_label(q)
# print(labelA)

# ____________________________________________________

def save_images(image, size, path):
    return imsave(inverse_transform(image), size, path)

def imsave(image, size, path):
    return scipy.misc.imsave(path, merge(image, size), format='png')

def merge(image, size):
    # print(type(image))
    [n, h, w, c] = image.shape
    # print(n, h, w, c)
    image = image.reshape(n * h, w, c).astype(np.float)
    if c == 1:
        image = image.reshape(n * h, w)
    img = image[:h * size[0]]
    # print(img.shape)
    # print(size[1])
    for i in range(size[1] - 1):
        img = np.concatenate((img, image[(i + 1) * h * size[0]:(i + 2) * h * size[0]]), axis=1)
    # print(img.shape)
    # print(img.shape)
    return img

def inverse_transform(image):
    return (image+1.)/2.



# if __name__ == '__main__':
#
#     pass


# python svhnmat.py