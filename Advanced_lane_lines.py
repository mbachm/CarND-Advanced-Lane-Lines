import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import camera_calibration
import color_gradient_threshold
import perspective_transform
import histogram

def pipeline(images):
	""" Pipeline for finding lane lines. It needs a array of the images to proceed.
	The pipeline consists of 5 Steps:
		1) Camera Calibration (done once)
		2) Distortion correction (done for each image)
		3) Color & Gradient Threshold (done for each image)
		4) Perspective Transform (done for each image)
		5) Peaks in Histogram (done for each image)
	"""
	### Step 1
	mtx, dist = camera_calibration.get_camera_calibration_values()
	for fname in images:
		img = mpimg.imread(fname)

		### Step 2
		dst = cv2.undistort(img, mtx, dist, None, None)

		### Step 3
		combined = color_gradient_threshold.apply_thresholds(dst)

		### Step 4
		mask = perspective_transform.apply_standard_mask_to_image(combined)
		warped = perspective_transform.warp(mask)
		ploty, left_fitx, right_fitx, left_fit, right_fit, left_curverad, right_curverad, histogram_image = histogram.histogram(warped)
		filled_image = histogram.fillPolySpace(histogram_image, left_fitx, right_fitx, ploty)
		#rewarp image
		rewarped = perspective_transform.warp(histogram_image, toBirdView=False)

		#stack images
		result = perspective_transform.weighted_img(dst, rewarped, α=0.8, β=1, λ=0)

		# add text
		font = cv2.FONT_HERSHEY_SIMPLEX
		curvature = 0.5*(round(left_curverad/1000,1) + round(right_curverad/1000,1))
		curvature_text = str('radius of curvature: '+str(curvature)+'km')
		cv2.putText(result, curvature_text, (430,630), font, 1, (0,0,255), 2, cv2.LINE_AA)
		# add text for distance from center
    	# 

		f, ((ax11, ax12, ax13, ax14),(ax21, ax22, ax23, ax24)) = plt.subplots(2, 4, figsize=(24, 9))
		f.tight_layout()
		ax11.imshow(img)
		ax11.set_title('Original Image', fontsize=50)
		ax12.imshow(dst)
		ax12.set_title('Undistorted Image', fontsize=50)
		ax13.imshow(combined, cmap='gray')
		ax13.set_title('Combination', fontsize=50)
		ax14.imshow(mask, cmap='gray')
		ax14.set_title('masked image', fontsize=50)
		ax21.imshow(warped, cmap='gray')
		ax21.set_title('warped image', fontsize=50)
		ax22.imshow(histogram_image)
		ax22.plot(left_fitx, ploty, color='yellow')
		ax22.plot(right_fitx, ploty, color='yellow')
		ax22.set_title('histogram image', fontsize=50)
		ax23.imshow(rewarped)
		ax23.set_title('rewarped image', fontsize=50)
		ax24.imshow(result)
		ax24.set_title('final image', fontsize=50)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.savefig('./lane_lines_images/'+fname.split('/')[-1], dpi=100)

test_images = glob.glob('./test_images/test*.jpg')
pipeline(test_images)
