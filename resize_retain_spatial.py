import numpy as np
import os, sys
from PIL import Image 
import cv2
import imutils
import math 

def resize_numpy(im_path, patch_size, show_im=False):
	'''
	*resizes images in numpy array form without altering spatial information; i.e. photo does not become "fatter" or "thinner"
	*if h,w proportion of resized im is very different from original, new im will have a lot of black edges
	*essentially creates a black background of desired dimensions and pastes resized photo onto it

	im_path: image file name/path
	patch_size: dimensions of output, tuple of height, width, channels; e.g. (256,256,3)

	'''
	
	im = cv2.imread(im_path)[:,:,::-1] #RGB/BGR
	background = np.zeros(shape=patch_size, dtype=np.int32)
	if np.argmax(im.shape) == 1: #if width is greater, set width as base
		resize_im = imutils.resize(im, width=patch_size[1])
		shift = patch_size[0] - resize_im.shape[0]
		background[int(shift/2):resize_im.shape[0]+int(shift/2),:,:] = resize_im

	else:
		resize_im = imutils.resize(im, height=patch_size[0])
		shift = patch_size[1] - resize_im.shape[1]
		background[:,int(shift/2):resize_im.shape[1]+int(shift/2),:] = resize_im

	if show_im:
		import matplotlib.pyplot as plt
		plt.imshow(background)

	return background


def resize_pillow(im_path, patch_size, resample=Image.LANCZOS, show_im=False):
    """
    *adapted from https://github.com/VingtCinq/python-resize-image/blob/master/resizeimage/resizeimage.py#L98
    *resizes images in numpy array form without altering spatial information; i.e. photo does not become "fatter" or "thinner"
	*if h,w proportion of resized im is very different from original, new im will have a lot of black edges
	*essentially creates a black background of desired dimensions and pastes resized photo onto it

	im_path: image file name/path
	patch_size: dimensions of output, tuple of height, width; e.g. (256,256)

    """
	image = Image.open(im_path)
	#print(img_format, image.format)
	img = image.copy()
	img.thumbnail((patch_size[1], patch_size[0]), resample)
	background = Image.new('RGB', (patch_size[1], patch_size[0]), (0,0,0))
	img_position = (
	    int(math.ceil((patch_size[1] - img.size[1]) / 2)),
	    int(math.ceil((patch_size[0] - img.size[0]) / 2))
	)
	background.paste(img, img_position)
	background.format = image.format

    if show_im:
		import matplotlib.pyplot as plt
		plt.imshow(background)
    
    return background.convert('RGB')
    