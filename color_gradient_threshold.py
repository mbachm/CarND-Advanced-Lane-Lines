import numpy as np
import cv2

def __calculate_scaled_sobel(sobel_value):
    """ Calculates scaled sobel for the given sobel value. """
    return np.uint8(255*sobel_value/np.max(sobel_value))

def __abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    """ Calculates absolute sobel for a given image. The 'x' or 'y' orientation are
        optional, the default value is 'x'. Furthermore, you can optionally calibrate
        the kernel size and threshold values. The threshold defines if a white pixel 
        is drawn at that point or not. Returns a binary image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    scaled_sobel = __calculate_scaled_sobel(abs_sobel)
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def __mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    """ Calculates the magnitude of a given image. Kernel size and thresholds are
        optional. The threshold defines if a white pixel is drawn at that point or
        not. Returns a binary image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) 
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = __calculate_scaled_sobel(abs_sobelxy)
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary

def __dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ Calculates the direction of the gradient of a given image. Kernel size and 
        thresholds are optional. The threshold defines if a white pixel is drawn 
        at that point or not. Returns a binary image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return dir_binary

def __hls_threshold(image, thresh=(90, 255)):
    """ Calculates the direction of the gradient of a given image. Kernel size and 
        thresholds are optional. The threshold defines if a white pixel is drawn 
        at that point or not. Returns a binary image.
    """
    """hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > thresh[0]) & (s_channel < thresh[1])] = 1
    return s_binary

def __apply_white_yellow_threshold(image):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)

    ### White color thresholds
    lower_white = np.array([0,210,0], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    ### Yellow color thresholds
    lower_yellow = np.array([10,0,100], dtype=np.uint8)
    upper_yellow = np.array([30,220,255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    ### Combine colors
    combined_binary = np.zeros_like(white_mask)
    combined_binary[((white_mask == 255) | (yellow_mask == 255))] = 255
    combined_binary[(combined_binary == 255)] = 1
    return combined_binary

def apply_thresholds(image):
    """ Applies sobel for x orientation, magnitude, S-channel thresholds and a mask
        for the colors 'yellow' and 'white' on an given image and combines the results
        in a binary image which is returned. 
    """
    gradx = __abs_sobel_thresh(image, orient='x', thresh=(20, 100))
    #grady = __abs_sobel_thresh(image, orient='y', thresh=(20, 100))
    ksize = 31
    mag_binary = __mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50, 130))
    #dir_binary = __dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    hls_binary = __hls_threshold(image, thresh=(170, 255))
    yellow_white = __apply_white_yellow_threshold(image)

    combined = np.zeros_like(hls_binary)
    combined[(yellow_white == 1) | ((hls_binary == 1) & (gradx == 1) & (mag_binary == 1))] = 1
    return combined
