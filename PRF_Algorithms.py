import  csv
import numpy as np
import numpy.ma as ma
from numpy.linalg import eig, inv
from math import factorial
from scipy import optimize
from scipy.interpolate import griddata as sgriddata


class PRF_Algo(object):
        
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
