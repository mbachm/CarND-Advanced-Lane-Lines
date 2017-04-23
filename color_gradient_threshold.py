import numpy as np
import cv2

def __calculate_scaled_sobel(sobel_value):
    """ Calculates scaled sobel for the given sobel value. """
    return np.uint8(255*sobel_value/np.max(sobel_value))

def __abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
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
    dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return dir_binary

def apply_thresholds(image):
    """ Applies sobel for x and y orientation, magnitude and direction thresholds
        for an given image and combines the results in a binary image, which is 
        returned. 
    """
    ksize = 15
    gradx = __abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = __abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = __mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = __dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined