import numpy as np
import cv2
import matplotlib.pyplot as plt
import lane
from scipy import signal

# Choose the number of sliding windows
nwindows = 9
#Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50

def __calculate_center_of_image(left_line_post, right_line_post, width):
    """ Calculates the distance to the center of the image. """
    center_distance = (left_line_post+right_line_post)/2
    return center_distance

def __window_search(binary_warped, nonzero, forLeftLane=True):
    """ Performs a sliding window search with the help of a histogram.
        'binary_warped': The binary_warped the image.
        'nonzero': All x and y positions of all nonzero pixels in the image.
        'forLeftLane': Indicates if the search should be perfomed on the left side
        of the histogram or for the right side.
    """
    binary_warped_y_shape = binary_warped.shape[0]
    window_height = np.int(binary_warped_y_shape/nwindows)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped_y_shape/2):,:], axis=0)
    # Smoothen the histogram
    histogram_smooth = signal.medfilt(histogram, 15)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram_smooth.shape[0]/2)
    if forLeftLane:
        x_base = np.argmax(histogram_smooth[:midpoint])
    else:
        x_base = np.argmax(histogram_smooth[midpoint:]) + midpoint
    x_current = x_base

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped_y_shape - (window+1)*window_height
        win_y_high = binary_warped_y_shape - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_inds.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))
    return lane_inds

def __find_left_and_right_lane_indices(binary_warped, left_lane, right_lane):
    """ Finds the left and right line indices like in the udacity lesson. """
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    left_lane_inds = __window_search(binary_warped, nonzero, forLeftLane=True)#
    right_lane_inds = __window_search(binary_warped, nonzero, forLeftLane=False)
    return left_lane_inds, right_lane_inds

def find_lanes_with_histogram(binary_warped, left_lane, right_lane):
    """ Find lanes wit the help of a histogram. Draws the detected lanes
        onto the output image.
    """
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Find left and right lane indices
    left_lane_inds, right_lane_inds = __find_left_and_right_lane_indices(binary_warped, left_lane, right_lane)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # After x and y for left and right lane were found, update them
    left_detection, left_n_buffered = left_lane.update(leftx, lefty)
    right_detection, right_n_buffered = right_lane.update(rightx, righty)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    ### Calculate distance from center
    center_distance = __calculate_center_of_image(left_lane.line_base_pos, right_lane.line_base_pos, binary_warped.shape[1])
    return left_lane, right_lane, center_distance, out_img

def fillPolySpace(image, left_lane, right_lane):
    """ Fill the space between the found lanes. """
    pts_left = np.array([np.transpose(np.vstack([left_lane.bestx, left_lane.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.bestx, right_lane.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    return cv2.fillPoly(image, np.int_([pts]), (0,255, 0))
