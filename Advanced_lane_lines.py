import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import camera_calibration
import color_gradient_threshold
import perspective_transform
import find_lanes

def put_texts_on_image(image, curvature, center_distance):
	""" Uses cv2.putText to draw the values of curvature and center_distance
		onto to given image
	"""
	font = cv2.FONT_HERSHEY_SIMPLEX
	curvature_text = str('radius of curvature: '+str(curvature)+'km')
	cv2.putText(image, curvature_text, (430,630), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
	position_text = 'Vehicle position : %.2f m %s of center'%(abs(center_distance), 'left' if center_distance < 0 else 'right')
	cv2.putText(image,position_text,(430,670), font, 0.8,(0,0,255), 2, cv2.LINE_AA)
	return image


def pipeline(images, testMode=False):
	""" Pipeline for finding lane lines. It needs a array of the images to proceed.
	The pipeline consists of serveral steps:
		1) Camera Calibration (done once)
		2) Distortion correction (done for each image)
		3) Color & Gradient Threshold (done for each image)
		4) Perspective Transform (done for each image)
		5) Find lanes, calculate curvature and distance to center  with peaks in 
		   histogram. Draw found lanes in transformed image (done for each image)
		6) Retransform image and stack undistorted and rewarped image (done for each image)
		7) Write text with calculations of step 5 on image (done for each image)
	"""
	### Step 1
	mtx, dist = camera_calibration.get_camera_calibration_values()
	for fname in images:
		img = mpimg.imread(fname)
		### Just for safety I work on a copy of the image
		copy_img = np.copy(img)

		### Step 2
		dst = cv2.undistort(copy_img, mtx, dist, None, None)

		### Step 3
		combined = color_gradient_threshold.apply_thresholds(dst)

		### Step 4
		mask = perspective_transform.apply_standard_mask_to_image(combined)
		warped = perspective_transform.warp(mask)

		### Step 5
		ploty, left_fitx, right_fitx, curvature, center_distance, histogram_image = find_lanes.find_lanes_with_histogram(warped)
		filled_image = find_lanes.fillPolySpace(histogram_image, left_fitx, right_fitx, ploty)
		
		### Step 6
		rewarped = perspective_transform.warp(histogram_image, toBirdView=False)
		result = perspective_transform.weighted_img(dst, rewarped, α=0.8, β=1, λ=0)
		
		### Step 7
		result = put_texts_on_image(result, curvature, center_distance)

		### Plot result of each step to the output_images folder if run in test mode
		if testMode:
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
			plt.savefig('./output_images/'+fname.split('/')[-1], dpi=100)

test_images = glob.glob('./test_images/test*.jpg')
pipeline(test_images, testMode=True)
straight_lines_images = glob.glob('./test_images/straight_lines*.jpg')
pipeline(straight_lines_images, testMode=True)
