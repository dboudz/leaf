#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 22:49:18 2017

@author: dbo
"""

from __future__ import print_function
import os
from PIL import Image, ImageChops, ImageOps

#http://pillow.readthedocs.io/en/3.1.x/handbook/tutorial.html
#https://auth0.com/blog/image-processing-in-python-with-pillow/
#https://stackoverflow.com/questions/9103257/resize-image-maintaining-aspect-ratio-and-making-portrait-and-landscape-images-e


FOLDER_TRAINING_SET='/Users/dboudeau/depot/leaf/'
FARGESIA=os.listdir(FOLDER_TRAINING_SET+'FARGESIA/') 


def makeThumb(f_in, f_out, size=(400,400), pad=True):

    image = Image.open(f_in)
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size

    if pad:
        thumb = image.crop( (0, 0, size[0], size[1]) )
        offset_x = max( (size[0] - image_size[0]) / 2, 0 )
        offset_y = max( (size[1] - image_size[1]) / 2, 0 )
        thumb = ImageChops.offset(thumb, int(offset_x), int(offset_y))
    else:
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, (0.5, 0.5))

    thumb.save(f_out)

cmpt=0
for infile in FARGESIA:
    f, e = os.path.splitext(infile)
    outfile = f + ".jpg"
    try:
        image = Image.open(FOLDER_TRAINING_SET+'FARGESIA/'+infile)
        makeThumb(FOLDER_TRAINING_SET+'FARGESIA/'+infile, str(cmpt)+'.jpg')
        #image.thumbnail((400, 400))
        #image.save(str(cmpt)+'.jpg')
        #resize(image, (400,400), True, str(cmpt)+'.jpg')
        cmpt=cmpt+1
    except IOError as e:
        print(e)


