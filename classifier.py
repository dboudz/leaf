#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:54:09 2017

@author: dbo
"""

import os 
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

FOLDER_TRAINING_SET='/Users/dboudeau/depot/leaf/'

#THUYA=os.listdir(FOLDER_TRAINING_SET+'THUYA/') 
FARGESIA=os.listdir(FOLDER_TRAINING_SET+'FARGESIA/') 


#img = cv2.imread('/Users/dboudeau/depot/leaf/THUYA/290px-Thuja_standishii.jpg')
##imgr = cv2.imread(image)
#plt.imshow(img)



THUYA=['/Users/dboudeau/depot/leaf/THUYA/290px-Thuja_standishii.jpg']

# step 2
filename_queue = tf.train.string_input_producer(THUYA)

image_contents = tf.read_file('/Users/dboudeau/depot/leaf/THUYA/290px-Thuja_standishii.jpg')
image = tf.image.decode_jpeg(image_contents, channels=3)
print(image)
image = tf.image.resize_image_with_crop_or_pad(image, 250, 250)

with tf.Session() as sess:
  print(image.eval())
#image =read_image('/Users/dboudeau/depot/leaf/THUYA/290px-Thuja_standishii.jpg')

#resized_image = tf.image.resize_images(image, [224, 224])
