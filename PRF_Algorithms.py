"""
Module Name: PRF_Algorithms
Project: Data-to-Vision
Owner: Gary Davis
Class Description: This class contains the algorithms used by the analysis classes.
        It contains algos for LMS circular fit, LMS Plane fit, filter, etc.
"""

import  csv
import numpy as np
import numpy.ma as ma
from numpy.linalg import eig, inv
from math import factorial
from scipy import optimize
from scipy.interpolate import griddata as sgriddata


class PRF_Algo(object):
        #Circle Functions########################################################################################################################################################################### 
        def calc_R(self, x,y, xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x-xc)**2 + (y-yc)**2)
         
        def f(self, c, x, y):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = self.calc_R(x, y, *c)
            return Ri - Ri.mean()
         
        def leastsq_circle(self,x,y,offet=0):
            # coordinates of the barycenter
            x_m = np.mean(x)
            y_m = np.mean(y)
            center_estimate = x_m, y_m
            center, ier = optimize.leastsq(self.f, center_estimate, args=(x,y))
            xc, yc = center
            Ri       = self.calc_R(x, y, *center)
            MAX_Ri = np.max(Ri)
            MIN_Ri = np.min(Ri)
            R        = Ri.mean()
            residu   = np.sum((Ri - R)**2)
            theta_fit = np.linspace(-np.pi, np.pi, 180)
            x_fit = xc + (R+offet)*np.cos(theta_fit)
            y_fit = yc + (R+offet)*np.sin(theta_fit)
            C_Fit = (R+offet)*2
            Circularity = (MAX_Ri - MIN_Ri)
            return xc, yc, R, residu, MAX_Ri, MIN_Ri, x_fit, y_fit, Circularity, C_Fit
        
        #Least Squares Plane Fits############################################################################################################################################################################################
        def LMS_PlaneFit(self,XData_List,YData_List,ZData_List,mask,order=1):
            maxOrder = 4
            if order == 1:
                A = np.column_stack((np.ones(np.array(XData_List)[mask].size), np.array(XData_List)[mask], np.array(YData_List)[mask]))
                c, resid,rank,sigma = np.linalg.lstsq(A,np.array(ZData_List)[mask],rcond=None)
                NZ = (np.array(XData_List)*float(c[1]))+(np.array(YData_List)*float(c[2]))+c[0]
                NormZ = np.array(ZData_List) - NZ
                r2 = 1 - resid / (np.array(ZData_List)[mask].size * np.array(ZData_List)[mask].var())
                #print r2
                #print resid
                return NormZ


        def interpNaN(self, XX, YY, NArray, XVx, YVx):
            NArray = sgriddata((YY, XX), NArray, (YVx, XVx), method='nearest')
            return NArray

        #Find Nearest Value#############################################################################################################################################################################
        def find_nearest(self,array,value):
            idx = (np.abs(array-value)).argmin()
            return [array[idx], idx]


        #Filter Function################################################################################################################################################################################
        def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
            r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
            The Savitzky-Golay filter removes high frequency noise from data.
            It has the advantage of preserving the original shape and
            features of the signal better than other types of filtering
            approaches, such as moving averages techniques.
            Parameters
            ----------
            y : array_like, shape (N,)
                the values of the time history of the signal.
            window_size : int
                the length of the window. Must be an odd integer number.
            order : int
                the order of the polynomial used in the filtering.
                Must be less then `window_size` - 1.
            deriv: int
                the order of the derivative to compute (default = 0 means only smoothing)
            Returns
            -------
            ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
            Notes
            -----
            The Savitzky-Golay is a type of low-pass filter, particularly
            suited for smoothing noisy data. The main idea behind this
            approach is to make for each point a least-square fit with a
            polynomial of high order over a odd-sized window centered at
            the point."""
            try:
                window_size = np.abs(np.int(window_size))
                order = np.abs(np.int(order))
            except ValueError:
                raise ValueError("window_size and order have to be of type int")
            if window_size % 2 != 1 or window_size < 1:
                raise TypeError("window_size size must be a positive odd number")
            if window_size < order + 2:
                raise TypeError("window_size is too small for the polynomials order")
            order_range = range(order+1)
            half_window = (window_size -1) // 2
            # precompute coefficients
            b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
            m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
            # pad the signal at the extremes with
            # values taken from the signal itself
            firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
            lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))
            return np.convolve( m[::-1], y, mode='valid')

#Sort the data by height and fill with something################################################################################################################################################
        def SortAndFill(self, NArray, Depth, Fill, Compare):
            if Compare == 'G':
                UpdatedArray = ma.masked_where(np.array(NArray) > float(Depth), np.array(NArray))
                UpdatedArray = UpdatedArray.filled(Fill)
            if Compare == 'GE':
                UpdatedArray = ma.masked_where(np.array(NArray) >= float(Depth), np.array(NArray))
                UpdatedArray = UpdatedArray.filled(Fill)
            if Compare == 'L':
                UpdatedArray = ma.masked_where(np.array(NArray) < float(Depth), np.array(NArray))
                UpdatedArray = UpdatedArray.filled(Fill)
            if Compare == 'LE':
                UpdatedArray = ma.masked_where(np.array(NArray) <= float(Depth), np.array(NArray))
                UpdatedArray = UpdatedArray.filled(Fill)
            if Compare == 'Equal':
                UpdatedArray = ma.masked_where(np.array(NArray) == Depth, np.array(NArray))
                UpdatedArray = UpdatedArray.filled(Fill)
            return UpdatedArray



