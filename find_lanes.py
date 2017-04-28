import numpy as np
import cv2
import matplotlib.pyplot as plt

# Choose the number of sliding windows
nwindows = 9
#Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def __calculate_real_space_curvature(ploty, left_fit, right_fit, pixel_pos):
    """ Calculates the curvature in real space. """
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    ### Scaling in real world space
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(pixel_pos[0]*ym_per_pix, pixel_pos[1]*xm_per_pix, 2)
    right_fit_cr = np.polyfit(pixel_pos[2]*ym_per_pix, pixel_pos[3]*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curvature = 0.5*(round(left_curverad/1000,1) + round(right_curverad/1000,1))
    return curvature

def __calculate_center_of_image(left_fitx, right_fitx, width):
    """ Calculates the distance to the center of the image. """
    middle = (left_fitx[-1] + right_fitx[-1])//2
    veh_pos = width//2
    center_distance = (veh_pos - middle)*xm_per_pix
    return center_distance

def __find_left_and_right_lane_indices(binary_warped_y_shape, leftx_base, rightx_base, nonzero, nonzerox, nonzeroy):
    """ Finds the left and right line indices like in the udacity lesson. """
    # Set height of windows
    window_height = np.int(binary_warped_y_shape/nwindows)
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped_y_shape - (window+1)*window_height
        win_y_high = binary_warped_y_shape - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    return left_lane_inds, right_lane_inds

def find_lanes_with_histogram(binary_warped):
    """ Find lanes wit the help of a histogram. """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Find left and right lane indices
    left_lane_inds, right_lane_inds = __find_left_and_right_lane_indices(binary_warped.shape[0], leftx_base, rightx_base, nonzero, nonzerox, nonzeroy)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ### Visualization
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    ### Calculate curvature
    curvature = __calculate_real_space_curvature(ploty, left_fit, right_fit, [lefty, leftx, righty, rightx])
    ### Calculate distance from center
    center_distance = __calculate_center_of_image(left_fitx, right_fitx, binary_warped.shape[1])

    return ploty, left_fitx, right_fitx, curvature, center_distance, out_img

def fillPolySpace(image, left_fitx, right_fitx, ploty):
    """ Fill the space between the found lanes.
    """
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    return cv2.fillPoly(image, np.int_([pts]), (0,255, 0))
