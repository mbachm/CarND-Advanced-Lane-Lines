#Writeup Report
---

**Advanced Lane Finding Project**

[//]: # (Image References)

[dist]: ./camera_cal/calibration8.jgp "Distorted"
[undist]: ./output_images/calibration8.jpg "Undistorted"
[straight_lines1]: ./output_images/straight_lines1.jpg "Straight lines 1"
[straight_lines2]: ./output_images/straight_lines2.jpg "Straight lines 2"
[test1]: ./output_images/test1.jpg "Warp Example"
[test2]: ./output_images/test2.jpg "Fit Visual"
[test3]: ./output_images/test3.jpg "Output"
[test4]: ./output_images/test4.jpg "Warp Example"
[test5]: ./output_images/test5.jpg "Fit Visual"
[test6]: ./output_images/test6.jpg "Output"
[video1]: ./processed_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup Report

My project consists of multiple python modules: 
* `advanced_lane_lines.py`: The main class, where all other methods/modules are called from
* `camera_calibration.py`: The module for the camera calibration
* `color_gradient_threshold.py`: Module to create a binary thresholded image
* `perspective_transform.py`: Module to transform/warp images and for applying the region of interest image mask
* `find_lanes.py`: Find lanes wit the help of a histogram and curvature and the distance to the center of the car

In the main module `advanced_lane_lines.py` all important steps are called. The most important function is `pipeline()`(line 24-92). It consinst of 7 steps which generate my result:
1. Camera Calibration (done if necessary)
2. Distortion correction
3. Color & Gradient Threshold
4. Perspective Transform
5. Find lanes, calculate curvature and distance to center  with peaks in histogram. Draw found lanes in transformed image
6. Retransform image and stack undistorted and rewarped image
7. Write text with calculations of step 5 on image

The results of the test images where generated with the function `test_pipeline()` (line 95 -105), which calls `pipeline()`. For a better performance, the camera calibration values where calculated before calling `pipeline()`.

The code of lines 67-90 was only necessary to generate and save a nice image which contains all steps if my pipeline seperately. I will explain it any further.

The project video was computed with the function `video_pipeline()` (lines 107-109), which does not calculates the camera calibration before calling `pipeline()`. Instead, it reads the global variables `mtx` and `dist`. The values were calcluated in line 123 before processing the project video (line 123-128).

Now I will go through each necessary rubric point and explain the details of my implementation.

###Camera Calibration

The code for this step is contained in camera_calibration.py. `get_camera_calibration_values()` (line 54-60) and `get_camera_calibration_values_and_save_example_camera_calibration_images()` (line 44-52) are the public methods for this python module. The second was only used to save the undistort images, the first is used for the pipeline.

The code of `get_camera_calibration_values()` calls the function `__calibrate_camera()` (line 8-33), which is pretty the same as in the 'Correcting for Distortion' lesson. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  The object points have a size of 6*9 as the chessboards have 9 columns and 6 rows (line 16-20). Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. To find a chessboard on the image, I use the `cv2.findChessboardCorners()` function (line 23-30).

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function (line 32-33) and obtained this result: 

| Distored image | Undistored image |
|:---:|:---:|
| ![distorted image][dist] | ![undistorted image][undist] |

###Pipeline (single images)

| Pipeline steps for straight line images |
|:---:|
| ![straight line image 1][straight_lines1] |
| ![straight line image 2][straight_lines2] |

####Distortion-correction

You can see in the two images above under the headline 'Undistorted Image' that I applied the distortion correction to the straight line images. This was done in step 2 in my pipeline (`advanced_lane_lines.py`, line 47). I used the `cv2.undistort()` function with my previos calculated values to perform the distortion correction

####Color transforms, gradients or other methods to create a thresholded binary image.

The code for this can be found in the model `color_gradient_threshold.py`. I used a combination of absolute sobelx and S-channel thresholds to generate a binary image (function `apply_thresholds()` lines 71-85). To calculate the absolute x sobel threshold, I used a function of which was presented in a Udacity lession (`__abs_sobel_thresh()`, line 8-23). The HLS color transformation and threshold is performed in the function `__hls_threshold()` (lines 54-69). Both function generate a binary image, which is combines in the `apply_thresholds()` function (line 83-84).

I also experimented with direction, magnitude and sobely thresholds, as you can see in this module (lines 25-69 and 77-80). But the above combination provided the best results for me. You can the result of the `apply_thresholds()` function in the images above with the title 'Combination'.

This parted is called in `advanced_lane_lines.py` in step 3, line 50.

####Perspective transform

My implementation for the perspective transformation can be found in the module `perspective_transform.py`. The function `apply_standard_mask_to_image()` applies a mask to the image (see 'Masked Image' in the images above), which is called in line 53 in `advanced_lane_lines.py`. Afterwards, the function `warp()` is called, which takes an image and an optional boolean parameter `toBirdView`. There I define source and destination points for the transformation (lines 57 - 66 in `perspective_transform.py`). I chose the hardcode the source and destination points in the following manner:

```
def __standard_vertices_array(image, dtype=np.int32):
	""" Generates the standard vertices array for the image mask for the given image.
		This method optimized is for Carnd-term1 project 4 and assues that the image
		size is 1280x780.
	"""
	imshape = image.shape
	apex_width_offset = 500
	bottom_left = [bottom_width_offset, imshape[0]]
	bottom_right = [imshape[1]-bottom_width_offset, imshape[0]]
	apex_left = [apex_width_offset, apex_height]
	apex_right = [imshape[1]-apex_width_offset, apex_height]
	return np.array([bottom_left, bottom_right, apex_right, apex_left], dtype=dtype)

src = __standard_vertices_array(image, np.float32)


offset = 100
bottom_left = [np.float32(offset), np.float32(image.shape[0])]
bottom_right = [np.float32(image.shape[1]-offset), np.float32(image.shape[0])]
apex_left = [np.float32(offset), np.float32(0)]
apex_right = [np.float32(image.shape[1]-offset), np.float32(0)]

dst = np.array([bottom_left, bottom_right, apex_right, apex_left])
```

This resulted in the following source and destination points:

| Source    | Destination | 
|:---------:|:-----------:| 
| 100, 720  | 100, 720    | 
| 1180, 720 | 1180, 720   |
| 780, 480  | 1180, 0     |
| 500, 480  | 100, 0      |

Afterwars I use `cv2.getPerspectiveTransform()` to compute perspective transform matrix M and use M with `cv2.warpPerspective()` to create the warped/transformed image (lines 68-76 in `perspective_transform.py`). Depending of the boolean parameter toBirdView of the function `warp()` I decide if the transformation has to be done to bird view or back to normal (lines 69-73).

This parted is used in `advanced_lane_lines.py` in step 4 with toBirdView=True and 6 with toBirdView=False, line 53-54 and 61-62.

####Identifiy lane-line pixels and fit their positions with a polynomial

My implementation of this part can be found in the module `find_lanes.py`. The function `find_lanes_with_histogram()` (lines 68-117) does this. First, I take a histogram of the bottom half of the image (line 71). Then I find the peak of the left and right halves of the histogram, which should be the left and right lanes (line 77-79). Afterwars I identify the x and y positions of all nonzero pixels in the image (line 82-84) and find the left and right lane indices with the function `__find_left_and_right_lane_indices()` (line 87, function 36-66). As the code of the function is take from a Udacity lession, I will not explain it further.

Afterwards I concatenate the arrays, extract left and right line pixel positions and fit a second order polynomial to each (line 90-101).

The result of this step can be seen in the images above under the headline 'Identified Lanes', where I also filled the space between the two polynomials green (`advanced_lane_lines.py` step 5, lines 61 and 62).

####Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

This was also done in the function `find_lanes_with_histogram()` of the module `find_lanes.py` (line 113-115). To calculate the curvature, I call the function `__calculate_real_space_curvature()` of the same module. There I take the maximun y value and use the methods explained in the 'Measuring Curvature' lesson.

The calculation of the postion of the vehicle is done in the method `__calculate_center_of_image()` (line 32-37) and is simple.

```
def __calculate_center_of_image(left_fitx, right_fitx, width):
    """ Calculates the distance to the center of the image. """
    middle = (left_fitx[-1] + right_fitx[-1])//2
    veh_pos = width//2
    center_distance = (veh_pos - middle)*xm_per_pix
    return center_distance
```

First I calculate the center/middle of the two polynomials (left_fitx, right_fitx). Then I calculate the postion of the vehicle, which is the center of the image. Afterwards, I substract the middle of the vehicle postion and multiplies it with meters per pixel in x dimension `xm_per_pix = 3.7/700`.

####Results plotted back down onto the road with clearly identified lane area

This is done in step 7 of my pipeline (line 65, `advanced_lane_lines.py`). The code calls `__put_texts_on_image()` with the before calculated radius of curvature and the position of the vehicle (lines 13-22). The result of this step can be seen under the headline 'Final Image' in the provide images.

###Result of the single test images

Here are the other processed test images.

| Pipeline steps of the test images |
|:---:|
| ![test image 1][test1] |
| ![test image 2][test2] |
| ![test image 3][test3] |
| ![test image 4][test4] |
| ![test image 5][test5] |
| ![test image 6][test6] |

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

