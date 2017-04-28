import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

def __calibrate_camera(images, testMode=False):
	""" Reads in images for a given imagepath array and finds chessboard corners
	    for each image. Save the found corners in an array and the real world
		object points in another. After all images are processed, it returns the
		camera matrix and distortion coefficients of cv2.calibrateCamera as they
		are needed for the camera calibration.
	"""
	### Arrays to store object points and image points from all the images
	objpoints = [] # 3D points in real world space
	imgpoints = [] # 2D points in image plane
	### Preprare object points, like (0,0,0), (1,0,0), ...
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates

	### Read in each image and process is if possible
	for fname in images:
		img = mpimg.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
		### If corners are found, add object points, image points
		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)
	
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	return mtx, dist

def __save_undistorted_example_camera_calibration_images(images, mtx, dist):
	""" Undistort and save all images given in the images path array images
		to the folder 'output_images'
	"""
	for fname in images:
		img = mpimg.imread(fname)
		dst = cv2.undistort(img, mtx, dist, None, None)
		plt.imsave('./output_images/'+fname.split('/')[-1], dst)

def get_camera_calibration_values():
	""" Returns the camera matrix and distortion coefficients calculated of all
		calibration images of the folder './camera_cal/' with a name like
		calibration*.jpg.
		The calibration images have to be 9x6 chessboard images, otherwise this
		function will not work.
	"""
	calibration_images = glob.glob('./camera_cal/calibration*.jpg')
	return __calibrate_camera(calibration_images)

def get_camera_calibration_values_and_save_example_camera_calibration_images():
	""" Does the same as get_camera_calibration_values, but additionally saves
		the undistorted calibration images in the folder './camera_cal_undist/'
	"""
	calibration_images = glob.glob('./camera_cal/calibration*.jpg')
	mtx, dist = __calibrate_camera(calibration_images, testMode=True)
	__save_undistorted_example_camera_calibration_images(calibration_images, mtx, dist)
	return mtx, dist
