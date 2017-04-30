import numpy as np
from collections import deque

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

### Class to receive the characteristics of each line detection
class Lane:
    def __init__(self, queue_length=7, binary_warped_shape=720):
        # length of queue to store data
        self.queue_length = queue_length
        #number of fits in buffer
        self.n_buffered = 0
        # was the line detected in the last iterations?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = deque([],maxlen=queue_length)
        # second order polynomial of the last n fits of the line
        self.recent_fit = deque([],maxlen=queue_length)
        # real space x values of the last n fits of the line
        self.recent_rs_xfitted = deque([],maxlen=queue_length)

        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # real space average x values of the fitted line over the last n iterations
        self.rs_bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # x values of the most recent fit
        self.current_xfitted = [np.array([False])]
        # real space x values of the most recent fit
        self.current_rs_xfitted = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        self.real_space_radius_of_curvature = None

        self.ploty = np.linspace(0, binary_warped_shape-1, binary_warped_shape )
        
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def set_allxy(self, allx, ally):
        """ Sets allx and ally for further calculations. """
        self.allx = allx
        self.ally = ally

    def set_current_fit(self):
        """ Calculates the second order polynomial. """
        self.current_fit = np.polyfit(self.ally, self.allx, 2)

    def set_current_rs_xfit(self):
        """ Calculates the second order polynomial in the real space. """
        self.current_rs_xfitted = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)

    def set_current_xfitted(self):
        """ Calculates the values for plotting. """
        yvals = self.ploty
        self.current_xfitted = self.current_fit[0]*yvals**2 + self.current_fit[1]*yvals + self.current_fit[2]

    def set_bestx(self):
        """ Calculates average of the `recent_xfitted` values for smoother plotting. """
        fits = self.recent_xfitted
        if len(fits)>0:
            avg=0
            for fit in fits:
                avg +=np.array(fit)
            avg = avg / len(fits)
            self.bestx = avg

    def set_rs_bestx(self):
        """ Calculates average of the `recent_xfitted` values for smoother plotting. """
        fits = self.recent_rs_xfitted
        if len(fits)>0:
            avg=0
            for fit in fits:
                avg +=np.array(fit)
            avg = avg / len(fits)
            self.rs_bestx = avg

    def add_data(self):
        """ Adds the current found data to our buffer. """
        self.recent_xfitted.appendleft(self.current_xfitted)
        self.recent_fit.appendleft(self.current_fit)
        self.recent_rs_xfitted.appendleft(self.current_rs_xfitted)
        assert len(self.recent_xfitted)==len(self.recent_fit)
        self.n_buffered = len(self.recent_xfitted)

    def pop_data(self):
        """ Pops the latest data. """
        if self.n_buffered>0:
            self.recent_xfitted.pop()
            self.recent_fit.pop()
            self.recent_rs_xfitted.pop()
            assert len(self.recent_xfitted)==len(self.recent_fit)
            self.n_buffered = len(self.recent_xfitted)
        
        return self.n_buffered

    def set_best_fit(self):
        """ Calculates average of the `recent_fit` values for smoother a polynomial. """
        coeffs = self.recent_fit
        if len(coeffs)>0:
            avg=0
            for coeff in coeffs:
                avg +=np.array(coeff)
            avg = avg / len(coeffs)
            self.best_fit = avg

    def set_radius_of_curvature(self):
        """ Calculates radius of curvature in pixel and real space. """
        y_eval = max(self.ploty)
        if self.best_fit is not None:
            self.radius_of_curvature = ((1 + (2*self.best_fit[0]*y_eval + self.best_fit[1])**2)**1.5) /np.absolute(2*self.best_fit[0])
            self.real_space_radius_of_curvature = ((1 + (2*self.rs_bestx[0]*y_eval*ym_per_pix + self.rs_bestx[1])**2)**1.5) / np.absolute(2*self.rs_bestx[0])

    def get_diffs(self):
        """ Calculates the difference between the latest fit and the average for recent_fit. """
        if self.n_buffered>0:
            self.diffs = self.recent_fit - self.best_fit
        else:
            self.diffs = np.array([0,0,0], dtype='float')

    def check_detected_lane(self):
        """ Sanity check for the latest detected lane. """
        flag = True      
        if(self.n_buffered > 0):
            relative_delta = self.diffs / self.best_fit
            # allow maximally this percentage of variation in the fit coefficients from frame to frame
            if not (abs(relative_delta)<np.array([0.7,0.5,0.15])).all():
                flag=False
                
        return flag

    def update(self, allx, ally):
        """ Starts update of all calculations with a sanity check for the lane. """
        self.set_allxy(allx, ally)
        self.set_current_fit()
        self.set_current_xfitted()
        self.set_current_rs_xfit()
        self.get_diffs()
        if self.check_detected_lane():
            self.detected=True
            self.add_data()
            self.set_bestx()
            self.set_rs_bestx()
            self.set_best_fit()
        else:
            self.detected=False            
            self.pop_data()
            if self.n_buffered>0:
                self.set_bestx()
                self.set_rs_bestx()
                self.set_best_fit()

        self.set_radius_of_curvature()
        return self.detected, self.n_buffered
