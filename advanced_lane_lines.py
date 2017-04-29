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
import lane
from moviepy.editor import VideoFileClip

def __put_texts_on_image(image, left_lane, right_lane, center_distance):
	""" Uses cv2.putText to draw the values of curvature and center_distance
		onto to given image
	"""
	font = cv2.FONT_HERSHEY_SIMPLEX
	if left_lane.real_space_radius_of_curvature and right_lane.real_space_radius_of_curvature:
		curvature = 0.5*(round(right_lane.real_space_radius_of_curvature/1000,1) + round(left_lane.real_space_radius_of_curvature/1000,1))
		curvature_text = str('radius of curvature: '+str(curvature)+'km')
		cv2.putText(image, curvature_text, (430,630), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
	else:
		cv2.putText(image, 'Unable to calculate curvature', (430,630), font, 0.8, (0,0,255), 2, cv2.LINE_AA)

	if center_distance:
		position_text = 'Vehicle position : %.2f m %s of center'%(abs(center_distance), 'left' if center_distance < 0 else 'right')
		cv2.putText(image,position_text,(430,670), font, 0.8,(0,0,255), 2, cv2.LINE_AA)
	else:
		cv2.putText(image,'Unable to calculate vehicle position',(430,670), font, 0.8,(0,0,255), 2, cv2.LINE_AA)
	return image

def pipeline(img, left_lane=None, right_lane=None ,mtx=None, dist=None, fname=None, testMode=False):
	""" Pipeline for finding lane lines on a given image. If camera calibration was not done
		in a previous step and the result are not given as mtx and dist to the function, it
		will calibrate the camera on itself.

		The pipeline consists of serveral steps:
		1) Camera Calibration (done if necessary)
		2) Distortion correction
		3) Color & Gradient Threshold
		4) Perspective Transform
		5) Find lanes, calculate curvature and distance to center  with peaks in 
		   histogram. Draw found lanes in transformed image
		6) Retransform image and stack undistorted and rewarped image
		7) Write text with calculations of step 5 on image
		
		Furthermore, it will save the images to output_images folder if it is run in test mode.
	"""
	### Prepare lanes if function was called without
	if left_lane is None:
		left_lane = lane.Lane()
	if right_lane is None:
		right_lane = lane.Lane()

	### Step 1
	if mtx is None or dist is None:
		mtx, dist = camera_calibration.get_camera_calibration_values()

	### Step 2
	dst = cv2.undistort(img, mtx, dist, None, None)

	### Step 3
	combined = color_gradient_threshold.apply_thresholds(dst)

	### Step 4
	mask = perspective_transform.apply_standard_mask_to_image(combined)
	warped = perspective_transform.warp(mask)

	### Step 5
	left_lane, right_lane, center_distance, identified_lanes_image = find_lanes.find_lanes_with_histogram(warped, left_lane, right_lane)
	filled_image = find_lanes.fillPolySpace(identified_lanes_image, left_lane, right_lane)

	### Step 6
	rewarped = perspective_transform.warp(identified_lanes_image, toBirdView=False)
	result = perspective_transform.weighted_img(dst, rewarped, α=0.8, β=1, λ=0)

	### Step 7
	result = __put_texts_on_image(result, left_lane, right_lane, center_distance)

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
		ax14.set_title('Masked Image', fontsize=50)
		ax21.imshow(warped, cmap='gray')
		ax21.set_title('Warped Image', fontsize=50)
		ax22.imshow(identified_lanes_image)
		ax22.plot(left_lane.current_fit, left_lane.ploty, color='yellow')
		ax22.plot(right_lane.current_fit, right_lane.ploty, color='yellow')
		ax22.set_title('Identified Lanes', fontsize=50)
		ax23.imshow(rewarped)
		ax23.set_title('Rewarped Image', fontsize=50)
		ax24.imshow(result)
		ax24.set_title('Final Image', fontsize=50)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.savefig('./output_images/'+fname.split('/')[-1], dpi=100)

	return result


def test_pipeline(images):
	""" Pipeline for finding lane lines. It needs a array of the images to proceed.
		The function calls pipeline() to proceed each image.
	"""
	### Calculate calibration once
	mtx, dist = camera_calibration.get_camera_calibration_values()
	for fname in images:
		img = mpimg.imread(fname)
		### Just for safety I work on a copy of the image
		copy_img = np.copy(img)
		pipeline(img, mtx=mtx, dist=dist, fname=fname, testMode=True)

def video_pipeline(image):
	global mtx, dist, left_lane, right_lane
	return pipeline(image, left_lane, right_lane, mtx, dist, testMode=False)

"""
camera_calibration.get_camera_calibration_values_and_save_example_camera_calibration_images()
"""


test_images = glob.glob('./test_images/test*.jpg')
test_pipeline(test_images)
straight_lines_images = glob.glob('./test_images/straight_lines*.jpg')
test_pipeline(straight_lines_images)


"""
mtx, dist = camera_calibration.get_camera_calibration_values()
left_lane = lane.Lane()
right_lane = lane.Lane()
output = 'processed_project_video.mp4'
clip = VideoFileClip('project_video.mp4')
output_clip = clip.fl_image(video_pipeline)
output_clip.write_videofile(output, audio=False)
"""
