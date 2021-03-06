import numpy as np
import cv2
import matplotlib.image as mpimg


def __standard_vertices_array(image, dtype=np.int32):
	""" Generates the standard vertices array for the image mask for the given image.
		This method optimized is for Carnd-term1 project 4 and assues that the image
		size is 1280x780.
	"""
	imshape = image.shape
	bottom_width_offset = 120
	apex_width_offset = 550
	apex_height = 445
	bottom_left = [bottom_width_offset, imshape[0]]
	bottom_right = [imshape[1]-bottom_width_offset, imshape[0]]
	apex_left = [apex_width_offset, apex_height]
	apex_right = [imshape[1]-apex_width_offset, apex_height]
	return np.array([bottom_left, bottom_right, apex_right, apex_left], dtype=dtype)

def __region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def apply_standard_mask_to_image(image):
	""" Generates the standard vertices array and apply it as a image mask to the 
		given image.
	"""
	vertices = [__standard_vertices_array(image)]
	return __region_of_interest(image, vertices)

def warp(image, toBirdView=True):
	""" Performs an image transformation and returns the warped image and the inverse
		perspective transformation matrix.
	"""
	img_size = (image.shape[1], image.shape[0])
	src = __standard_vertices_array(image, np.float32)

	offset = 100
	bottom_left = [np.float32(4*offset), np.float32(image.shape[0])]
	bottom_right = [np.float32(image.shape[1]-(4*offset)), np.float32(image.shape[0])]
	apex_left = [np.float32(offset), np.float32(0)]
	apex_right = [np.float32(image.shape[1]-offset), np.float32(0)]

	dst = np.array([bottom_left, bottom_right, apex_right, apex_left])

	# Compute the perspective transform, M
	if toBirdView:
		M = cv2.getPerspectiveTransform(src, dst)
	# Compute inverse perspective transform, Minv
	else:
		M = cv2.getPerspectiveTransform(dst, src)

	# Create warped image - uses linear interpolation
	warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
	return warped

def weighted_img(undst, rewarped, α=0.8, β=1., λ=0.):
	""" Calculates the weighted sum of two arrays / images.
	"""
	return cv2.addWeighted(undst, α, rewarped, β, λ)
