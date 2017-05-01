# Writeup Report
---

**Advanced Lane Finding Project**

[//]: # (Image References)

[dist]: ./camera_cal/calibration8.jpg "Distorted"
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
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup Report

My project consists of multiple python modules: 
* `advanced_lane_lines.py`: The main class, where all other methods/modules are called from
* `lane.py`: Module of a lane. It keeps track of recent found lanes and bufferes them for a smoother result
* `camera_calibration.py`: The module for the camera calibration
* `color_gradient_threshold.py`: Module to create a binary thresholded image
* `perspective_transform.py`: Module to transform/warp images and for applying the region of interest image mask
* `find_lanes.py`: Find lanes wit the help of a histogram and curvature and the distance to the center of the car

In the main module `advanced_lane_lines.py` all important steps are called. The most important function is `pipeline()`(line 33-106). It consinst of 7 steps which generate my result:
1. Camera calibration (done if necessary)
2. Distortion correction
3. Color & gradient thresholding
4. Perspective transformation
5. Find lanes, calculate curvature and distance to center  with peaks in histogram. Draw found lanes in transformed image
6. Retransform image and stack undistorted and rewarped image
7. Write text with calculations for curvature and vehicle postion on image

All important values of the found lanes are stored in lane objects (one for the left and one for the right lane). Inside the lane object, all needed calculations are performed (see `lane.py`).

The results of the test images where generated with the function `test_pipeline()` (line 95 -105), which calls `pipeline()`. The code of lines 82-104 was only necessary to generate and save a nice image which contains all steps if my pipeline seperately. I will not explain it.

The project video was computed with the function `video_pipeline()` (lines 121-123). The values for the camera calibration where generated in line 126 both for `test_pipeline()` and `video_pipeline()`. This way, they only had to be caclulated once.

Now I will go through each necessary rubric point and explain the details of my implementation.

### Camera Calibration

The code for this step is contained in camera_calibration.py. `get_camera_calibration_values()` (line 43-51) and `save_example_camera_calibration_images()` (line 53-58) are the public methods for this python module. The second was only used to save the undistort images, the first is used for the pipeline.

The code of `get_camera_calibration_values()` calls the function `__calibrate_camera()` (line 8-33), which is nearly the same as in the 'Correcting for Distortion' lesson. I start by preparing 'object points', which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  The object points have a size of 6*9 as the chessboards have 9 columns and 6 rows (line 16-20). Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. To find a chessboard on the image, I use the `cv2.findChessboardCorners()` function (line 23-30).

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function (line 32-33) and obtained this result: 

| Distored image | Undistored image |
|:---:|:---:|
| ![distorted image][dist] | ![undistorted image][undist] |

### Pipeline (single images)

| Pipeline steps for straight line images |
|:---:|
| ![straight line image 1][straight_lines1] |
| ![straight line image 2][straight_lines2] |

#### Distortion-correction

You can see in the two images above under the headline 'Undistorted Image' that I applied the distortion correction to the straight line images. This was done in step 2 in my pipeline (`advanced_lane_lines.py`, line 61). I used the `cv2.undistort()` function with my previos calculated values to perform the distortion correction

#### Color transforms, gradients or other methods to create a thresholded binary image.

The code for this can be found in the model `color_gradient_threshold.py`. I used a combination of absolute sobelx, S-channel, magnitude thresholds togehter with a mask for the colors 'yellow' and 'white' to generate a binary image (function `apply_thresholds()` lines 91-106). To calculate the absolute x sobel threshold, I used a function of which was presented in a Udacity lession (`__abs_sobel_thresh()`, line 8-23). The HLS color transformation and threshold is performed in the function `__hls_threshold()` (lines 54-69), while the magnitude threshold is calculated in `__mag_thresh()` (lines 25-37). The yellow and white color masks are implemented in the function `__apply_white_yellow_threshold()` (lines 71-89). All functions generate a binary image which is combines in the `apply_thresholds()` function (line 104-105).

I also experimented with directionand sobely thresholds, as you can see in this module (lines 39.52). But the above combination provided the best results for me. You can the result of the `apply_thresholds()` function in the images above with the title 'Combination'.

This parted is called in `advanced_lane_lines.py` in step 3, line 64.

#### Perspective transform

My implementation for the perspective transformation can be found in the module `perspective_transform.py`. The function `apply_standard_mask_to_image()` applies a mask to the image (see 'Masked Image' in the images above), which is called in line 67 in `advanced_lane_lines.py`. Afterwards, the function `warp()` is called, which takes an image and an optional boolean parameter `toBirdView`. There I define source and destination points for the transformation (lines 56 - 64 in `perspective_transform.py`). I chose the hardcode the source and destination points in the following manner:

```
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
| 120, 720  | 400, 720    | 
| 1160, 720 | 880, 720    |
| 730, 445  | 1180, 0     |
| 550, 445  | 100, 0      |

Afterwars I use `cv2.getPerspectiveTransform()` to compute perspective transform matrix M and use M with `cv2.warpPerspective()` to create the warped/transformed image (lines 66-74 in `perspective_transform.py`). Depending of the boolean parameter toBirdView of the function `warp()` I decide if the transformation has to be done to bird view or back to normal (lines 67-71).

This parted is used in `advanced_lane_lines.py` in step 4 with toBirdView=True and 6 with toBirdView=False, line 67-68 and 75-76.

#### Identifiy lane-line pixels and fit their positions with a polynomial

My implementation of this part can be found in the module `find_lanes.py`. The function `find_lanes_with_histogram()` (lines 72-105) implements the main part. First I create an output image and identfiy the nonzero values for x and y (lines 77-82). Afterwars I apply a window search with the function `__find_left_and_right_lane_indices()` which calls `__window_search()` for left and right lanes. `__window_search()` performs a sliding window search (lines 21-61) and returns the indices of the pixels of a lane line. Therefor I take a histogram of the bottom half of the image and smooths it with signal.medfilt of scipy (line 31-33). Then I find the peaks of the histogram, which should be lanes (line 37-42). Afterwars I identify the x and y positions of all nonzero pixels in the image (line 4-46) and perform a sliding window search (line 49-61). As the code of the function is take from a Udacity lession, I will not explain it further.

Afterwards I concatenate the arrays for each lane (lines 88-89), extract left and right line pixel positions (lines 91-95) and update the lane objects (line 98 and 99), which will fit a second order polynomial and perform all further calculations in the `update()` function in lanes.py (line 147-170), which will also verifiy that the found lane is ok.

The result of this step can be seen in the images above under the headline 'Identified Lanes', where I also filled the space between the two polynomials green (`advanced_lane_lines.py` step 5, lines 71 and 72).

#### Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

The position of the vehicle is done in the function `__calculate_center_of_image()` in `find_lanes.py`. 

```
def __calculate_center_of_image(left_line_post, right_line_post, width):
    """ Calculates the distance to the center of the image. """
    center_distance = (left_line_post+right_line_post)/2
    return center_distance
```

The identify the middle of the two lanes it adds the positions of each lane and divide that by 2. The lane postions are calculated in `lane.py` (lines 132-134). It takes the x values of the most recent fit and substracts 640, which is the half of the image.

The radius of curvature is calculated by each lane itself in `radius_of_curvature()` (lanes.py, lines 118-123).

#### Results plotted back down onto the road with clearly identified lane area

This is done in step 7 of my pipeline (line 79, `advanced_lane_lines.py`). The code calls `__put_texts_on_image()` with the before calculated radius of curvature and the position of the vehicle (lines 13-22). The result of this step can be seen under the headline 'Final Image' in the provide images.

### Result of the single test images

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

### Pipeline (video)

#### Final video output

Here's a [link to my video result](./processed_project_video.mp4)

---

### Discussion

#### Problems and issues I faced in my implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

First I don't perform a skip of sliding the windows when I previously found a lane. This would speed up my pipeline and could lead to a more robust detection for the beginning and ending of a lane.

Second could optimize the thresholds for the binary image. I don't use the direction of the gradient or the absolute sobel for y orientation. With a better combination of the thresholds i would be able to dected lanes even better. Furthermore the yellow and white color thresholds are optimized for this project. If the color of the street or even the lane changes, this could be a problem for my pipeline.

Another aspect could be the rejection of outliners. This could lead to a better calculation of the polynomial for each lane.

Besides the mask of my binary image is optimized for 'nearly' straight streets without hard bends. I think my pipeline will likely fail on a very curvy road.
curvy road will