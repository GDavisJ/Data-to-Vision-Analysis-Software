"""
Module Name: PRF_ROUGH
Project: Data-to-Vision
Owner: Gary Davis
Class Description: This class is used to analyze roughness data.
        Multiple roughness parameters will be reported such as Ra, Rv, Rp, Rt, Rq, etc.
"""

import time, datetime, csv, stat, os, os.path, socket, \
       sys, shutil, distutils.dir_util, distutils.file_util
import numpy as np
import numpy.ma as ma
from PRF_Algorithms import PRF_Algo
from matplotlib import pyplot as plt, cm, colors
from scipy.signal import argrelextrema
import scipy.ndimage as ndimage
import matplotlib.gridspec as gridspec


class PRF_ROUGHNESS(object):
        def __init__(self, fname, fpath, LMS_Fit=False, filtType='None'):
                self.FName = fname
                self.FPath = fpath
                self.Name = ''
                self.Meas_Mag = 0.0
                self.FOV_Mag = 0.0
                self.Mag_Val = 0.0
                self.Camera_Pix_Size = 9.9
                self.Meas_Pix = 0.0
                self.Wavelength = 0.0
                self.DateTimeStamp = ''
                self.PanelID = ''
                self.BINARY = ''
                self.PSide = ''
                self.Unit_X = ''
                self.Unit_Y = ''
                self.Xnom = ''
                self.Ynom = ''
                self.StageX_Y = ''
                self.OutputName = ''
                self.PlotTitle = ''


                self.DataArr = []
                self.X_Array = ''
                self.Y_Array = ''
                self.ModArr = []
                self.XV = []
                self.YV = []
                self.filtType = filtType
                self.LMS_Fit = LMS_Fit
                self.crossX = 0.0
                self.crossY = 0.0

                #Roughness Calc Vars
                self.Ra = 0.0
                self.Rp = 0.0
                self.Rv = 0.0
                self.Rt = 0.0
                self.Rq = 0.0
                self.Rku = 0.0
                self.Rsk = 0.0
                self.Mr1 = 0.0
                self.Mr2 = 0.0
                self.Rk = 0.0
                self.Rpk = 0.0
                self.Rvk = 0.0
                self.V1 = 0.0
                self.V2 = 0.0
                self.Rpm = 0.0
                self.Rvm = 0.0
                self.Rz = 0.0

                self.YSect_0 = 0.0
                self.YSect_100 = 0.0
                self.Rmr_Perc = []
                self.Rmr_Ht = []


                #Method to Open and process the file
                self.Process_ASCII_File()




        #Open and Process ASCII File################################################################################################################################################
        def Process_ASCII_File(self):
                self.NFName = self.FName.split('.asc')[0]
                NameSplit = self.NFName.split('_')
                Date = NameSplit[9].split('-')[1]+'/'+NameSplit[9].split('-')[0]+'/'+NameSplit[9].split('-')[2]
                TimeVar = NameSplit[10].split('.')[0]+':'+NameSplit[10].split('.')[1]+':'+NameSplit[10].split('.')[2]
                DateTimeStamp = Date+' '+TimeVar
                DateTimeWrongFormat = datetime.datetime.strptime(str(DateTimeStamp), "%m/%d/%Y %H:%M:%S")
                self.DateTimeStamp = str(DateTimeWrongFormat).split(" ")[0].split('-')[1] + "/" + str(DateTimeWrongFormat).split(" ")[0].split('-')[2] \
                                       + "/" + str(DateTimeWrongFormat).split(" ")[0].split('-')[0]+ " " + str(DateTimeWrongFormat).split(" ")[1]
                self.PanelID = self.FName.split('_')[0].split('-')[1]
                self.BINARY = self.IDBinary(self.PanelID)
                self.PSide = NameSplit[2]
                self.Unit_X = NameSplit[3]
                self.Unit_Y = NameSplit[4]
                self.Xnom = NameSplit[5]
                self.Ynom = NameSplit[6]
                self.StageX_Y = NameSplit[11]+'_'+NameSplit[12]
                self.OutputName = NameSplit[0] + '_' +NameSplit[8] + '_' +NameSplit[1] + '_' +NameSplit[2] + '_Summary.txt'
                self.PlotTitle = NameSplit[0] + '_' +NameSplit[8] + '_' +NameSplit[1] + '_' +NameSplit[2] + '_' +NameSplit[3] + '_' +NameSplit[4] + '_' +NameSplit[5] + '_' +NameSplit[6]
                

                IntensityFound = "False"
                Open_txt = open(self.FPath+self.FName, 'r')
                reader = csv.reader(Open_txt, delimiter=',', quoting=csv.QUOTE_NONE)
                for row in reader:

                    if row[0] == "TurretMag":
                            self.Meas_Mag = float(row[3])
                    if row[0] == "FOVMag":
                            self.FOV_Mag = float(row[3])
                    if row[0] == "Wavelength":
                            self.Wavelength = float(row[3])/1000.000
                    if "Intensity" in row:
                            IntensityFound = "True"
                    if IntensityFound != "True" and len(row) == 481:
                            self.DataArr.append(row)
                    if "RAW_DATA" in row:
                            IntensityFound = "False"
                Open_txt.close()

                #Get Calculate the Mag, pixel size, and X/Y coordinates (based on pixel size)
                self.Mag_Val = self.Meas_Mag * self.FOV_Mag
                self.Meas_Pix = self.Camera_Pix_Size/float(self.Mag_Val)
                self.X_Array = np.linspace(0,639*self.Meas_Pix,640)
                self.Y_Array = np.linspace(0,479*self.Meas_Pix,480)
                self.crossX = self.Y_Array[int(len(self.Y_Array)/2)]
                self.crossY = self.X_Array[int(len(self.X_Array)/2)]
                

                #Change the rows to columns (the incoming orientation is wrong)
                columns1 = [[row[col] for row in self.DataArr] for col in range(len(self.DataArr[0]))]
                columns1.pop(-1)
                XV, YV = np.meshgrid(self.X_Array, self.Y_Array)
                XVx = np.array(XV)
                YVx = np.array(YV)
                self.XV = XV.reshape(np.array(XV).size).tolist()
                self.YV = YV.reshape(np.array(YV).size).tolist()


                #There is missing data in some of the locations so they need to be filled with Nans
                Cleanup = self.SortAndFill(np.array(columns1), '', np.nan, 'Equal')
                self.DataArr = Cleanup.astype(float)*float(self.Wavelength)
                self.updateData()



        #Method used to remove the tilt from the dataset.
        def tipTiltRemove(self):
                if self.LMS_Fit == True:
                        ZV = np.array(self.ModArr).reshape(np.array(self.ModArr).size).tolist()
                        #mask used so we can ignore the locations with Nans
                        mask =~np.isnan(ZV)

                        #Fit the 3D data plane and normalize the data
                        ZV = np.array(ZV).astype(float)
                        self.ModArr = PRF_Algo().LMS_PlaneFit(self.XV,self.YV,ZV,mask,1)
                        self.ModArr = np.array(self.ModArr)
                        self.ModArr.shape=(len(self.Y_Array),len(self.X_Array))
                        
                elif self.LMS_Fit == False:
                        self.ModArr = self.ModArr - np.nanmean(self.ModArr)

        #Method used to process the data using a Gaussian Filter
        def GausFilt(self):
                
                if self.filtType == 'Gaussian Low Pass':
                        self.ModArr = self.interpNaN(self.ModArr)
                        self.ModArr = ndimage.gaussian_filter(self.ModArr, sigma=2.0, order=0) #Low Pass Filter only

                elif self.filtType == 'Gaussian High Pass':
                        self.ModArr = self.interpNaN(self.ModArr)
                        LowPass = ndimage.gaussian_filter(self.ModArr, sigma=2.0, order=0) #Low Pass Filter only
                        self.ModArr = self.ModArr - LowPass



        def analysisChange(self, LMS_Fit=False, filtType='None'):
                self.filtType = filtType
                self.LMS_Fit = LMS_Fit
                self.updateData()
                
        #used to update the data based on user input.
        def updateData(self):
                self.ModArr = np.array(self.DataArr)
                self.tipTiltRemove()
                self.GausFilt()
                self.roughAnalysis()





        #Open and Process Via ASCII File################################################################################################################################################

        def roughAnalysis(self):
                mask =~np.isnan(self.ModArr)
                self.Ra = np.average(np.abs(self.ModArr[mask]))
                print ("Ra = " + str(self.Ra*1000) + "nm")
                self.Rp = np.max(self.ModArr[mask])
##                print ("Rp = " + str(self.Rp*1000) + "nm")
                self.Rv = np.min(self.ModArr[mask])
##                print ("Rv = " + str(self.Rv*1000) + "nm")
                self.Rt = np.max(self.ModArr[mask]) - np.min(self.ModArr[mask])
##                print ("Rt = " + str(self.Rt*1000) + "nm")
                self.Rq = np.sqrt(np.sum(self.ModArr[mask]**2)/(self.ModArr[mask].size))
                self.Rku = np.sum((self.ModArr[mask])**4)/(self.ModArr[mask].size*self.Rq**4)
                self.Rsk = np.sum((self.ModArr[mask])**3)/(self.ModArr.size*self.Rq**3)
                print ("Rku = " + str(self.Rku))
                print ("Rq = " + str(self.Rq*1000) + "nm")
                print ("Rsk = " + str(self.Rsk))


                self.Rmr_Perc = []
                self.Rmr_Ht = []
                PercVal = 0
                NPSort = self.ModArr[mask].reshape(np.array(self.ModArr[mask]).size).tolist()
                NPSort.sort()
                for Perc in range(1000):
                    self.Rmr_Perc.append(PercVal)
                    #Rmr_Ht.append(np.percentile(np.array(NPSort),PercVal))
                    PercVal = PercVal + 0.1            
                self.Rmr_Ht = np.percentile(np.array(NPSort),self.Rmr_Perc)
                self.Rmr_Perc.reverse()


                ZeroPerc = 0
                FrtyPerc = 400
                SmallestSecant = ''
                SectSlope = ''
                IndexVal_1 = ''
                IndexVal_2 = ''
                for FortyPerc in range(600):
                    SecantSlope = np.abs((float(self.Rmr_Ht[ZeroPerc])-float(self.Rmr_Ht[FrtyPerc]))/(float(self.Rmr_Perc[ZeroPerc])-float(self.Rmr_Perc[FrtyPerc])))
                    if SmallestSecant == '' or float(SecantSlope) <= float(SmallestSecant):
                        SectSlope = SecantSlope
                        SmallestSecant = SecantSlope
                        IndexVal_1 = ZeroPerc
                        IndexVal_2 = FrtyPerc
                    ZeroPerc = ZeroPerc + 1
                    FrtyPerc = FrtyPerc + 1
        ##        print SmallestSecant
        ##        print IndexVal_1
        ##        print IndexVal_2
        ##        print Rmr_Ht[IndexVal_1]
        ##        print Rmr_Ht[IndexVal_2]
        ##        print Rmr_Perc[IndexVal_1]
        ##        print Rmr_Perc[IndexVal_2]
                self.YSect_0 = (((float(self.Rmr_Ht[IndexVal_1])-float(self.Rmr_Ht[IndexVal_2]))/(float(self.Rmr_Perc[IndexVal_1])-float(self.Rmr_Perc[IndexVal_2])))*(0 - self.Rmr_Perc[IndexVal_1])) + self.Rmr_Ht[IndexVal_1]
                self.YSect_100 = (((float(self.Rmr_Ht[IndexVal_1])-float(self.Rmr_Ht[IndexVal_2]))/(float(self.Rmr_Perc[IndexVal_1])-float(self.Rmr_Perc[IndexVal_2])))*(100 - self.Rmr_Perc[IndexVal_1])) + self.Rmr_Ht[IndexVal_1]
        ##        print YSect_0
        ##        print YSect_100
                #x = ((yz - yy)/m)+xx
                

                self.Mr1 = float(self.Rmr_Perc[PRF_Algo().find_nearest(self.Rmr_Ht,self.YSect_0)[1]])
                self.Mr2 = self.Rmr_Perc[PRF_Algo().find_nearest(self.Rmr_Ht,self.YSect_100)[1]]
        ##        Equiv_Mr1 = (((float(Rmr_Ht[IndexVal_1])-float(Rmr_Ht[IndexVal_2]))/(float(Rmr_Perc[IndexVal_1])-float(Rmr_Perc[IndexVal_2])))*(Mr1 - Rmr_Perc[IndexVal_1])) + Rmr_Ht[IndexVal_1]
        ##        print Equiv_Mr1
                self.Rk = self.YSect_0 - self.YSect_100
##                print ("Mr1 = " + str(self.Mr1) + "%")
##                print ("Mr2 = " + str(self.Mr2) + "%")
##                print ("Rk = " + str(self.Rk *1000) + "nm")
                Rpk = self.SortAndFill(np.array(self.ModArr), self.YSect_0, np.nan, 'L')
                Rpkmask =~np.isnan(Rpk)
                self.Rpk = (np.sum(Rpk[Rpkmask] - self.YSect_0)*2) / len(Rpk[Rpkmask])
                Rvk = self.SortAndFill(np.array(self.ModArr), self.YSect_100, np.nan, 'G')
                Rvkmask =~np.isnan(Rvk)
                self.Rvk = (np.sum(self.YSect_100 - Rvk[Rvkmask])*2) / len(Rvk[Rvkmask])
                self.V1 = (self.Mr1/200)*self.Rpk
                self.V2 = ((100 -self.Mr2)/200)*self.Rvk
##                print ("Rpk = " + str(self.Rpk * 1000) + "nm")
##                print ("Rvk = " + str(self.Rvk * 1000) + "nm")
##                print ("V1 = " + str(self.V1 * 1000) + "nm")
##                print ("V2 = " + str(self.V2 * 1000) + "nm")
                

                """Rz Calculations"""
                tempArray = np.sort(np.array(self.ModArr[mask]))
##                print('array size: ', len(tempArray))
##                print('num pts: ', int(len(tempArray)*0.05))


##                Rvm_List = tempArray[:int(len(tempArray)*0.05)]
##                Rpm_List = tempArray[::-1][:int(len(tempArray)*0.05)]


                Rvm_List = tempArray[:10]
                Rpm_List = tempArray[::-1][:10]
                

                self.Rpm = np.average(Rpm_List)
                self.Rvm = np.average(Rvm_List)
                self.Rz = np.average(Rpm_List)-np.average(Rvm_List)
##                print ("Rpm = " + str(self.Rpm*1000) + "nm")
##                print ("Rvm = " + str(self.Rvm*1000) + "nm")
##                print ("Rz = " + str(self.Rz*1000) + "nm")



                
        #Gets the figure object and saves if needed################################################################################################################################################
        def getFigObj(self, bgColor, saveFig=False):
                plt.close("all")
                if bgColor == '#292929':
                        plt.style.use('dark_background')
                else :
                        plt.style.use('default')
                plotFontSize = 12

                        
                #Create the figure and grid for the ojects
                fig2 = plt.figure(constrained_layout=True)
                fig2.set_facecolor(bgColor)
                spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
                f2_ax1 = fig2.add_subplot(spec2[0, 0])
                f2_ax2 = fig2.add_subplot(spec2[1, 0])
                f2_ax3 = fig2.add_subplot(spec2[0, 1])
                f2_ax4 = fig2.add_subplot(spec2[1, 1])

                #Contour Plot creation
                f2_ax1.patch.set_color('0')
                f2_ax1.set_aspect('equal', adjustable='box')
                contourf_ = f2_ax1.contourf(self.X_Array, self.Y_Array, self.ModArr , 60, cmap=plt.get_cmap('jet'))#, cmap=plt.cm.mpl)
                fig2.colorbar(contourf_, format='%.4f', ax=f2_ax1)
                f2_ax1.set_title("Contour", fontsize=plotFontSize)
                f2_ax1.set_xlabel("X", fontsize=plotFontSize)
                f2_ax1.set_ylabel("Y", fontsize=plotFontSize)
                txtPos = self.X_Array[-1]*-1.16239
                f2_ax1.text(txtPos, int(479*self.Meas_Pix), 'Total Mag:  ' + str(round(self.Mag_Val,1))+"x",fontsize=12)
                f2_ax1.text(txtPos, int(420*self.Meas_Pix), 'Pixel Size: ' + str(round(self.Meas_Pix,4))+"um",fontsize=12)
                f2_ax1.text(txtPos, int(340*self.Meas_Pix), 'Ra:         ' + str(round(self.Ra*1000,4))+"nm",fontsize=12)
                f2_ax1.text(txtPos, int(280*self.Meas_Pix), 'Rq:         ' + str(round(self.Rq*1000,4))+"nm",fontsize=12)
                f2_ax1.text(txtPos, int(220*self.Meas_Pix), 'Rku:        ' + str(round(self.Rku,4)),fontsize=12)
                f2_ax1.text(txtPos, int(160*self.Meas_Pix), 'Rsk:        ' + str(round(self.Rsk,4)),fontsize=12)


                #X profile Creation
                f2_ax3.plot(self.X_Array, self.ModArr[int(float(self.crossX)/self.Meas_Pix)], 'b')
                f2_ax3.set_title("X Profile", fontsize=plotFontSize)
                f2_ax3.set_xlabel("X", fontsize=plotFontSize)
                f2_ax3.set_ylabel("Z", fontsize=plotFontSize)
                f2_ax3.set_facecolor(bgColor)



                #Y profile Creation
                f2_ax4.plot(self.Y_Array, self.ModArr[:,int(float(self.crossY)/self.Meas_Pix)], 'r')
                f2_ax4.set_title("Y Profile", fontsize=plotFontSize)
                f2_ax4.set_xlabel("Y", fontsize=plotFontSize)
                f2_ax4.set_ylabel("Z", fontsize=plotFontSize)
                f2_ax4.set_facecolor(bgColor)



                #Bearing Ratio Curve
                f2_ax2.set_title('Bearing Ratio', fontsize=plotFontSize)
                f2_ax2.set_ylabel('Height', fontsize=plotFontSize)
                f2_ax2.set_xlabel('Percent Area', fontsize=plotFontSize)
                f2_ax2.plot(self.Rmr_Perc,self.Rmr_Ht,color='black',marker="",ms=6,lw=2)
                f2_ax2.plot([0,100],[self.YSect_0,self.YSect_100],color='red',marker="",ms=6,)
                f2_ax2.plot([self.Mr1,self.Mr1],[self.Rv,self.Rp],color='green',marker="",ms=6,)
                f2_ax2.plot([self.Mr2,self.Mr2],[self.Rv,self.Rp],color='green',marker="",ms=6,)

                f2_ax2.fill_between(self.Rmr_Perc, self.YSect_0, self.Rmr_Ht, where=self.Rmr_Ht > self.YSect_0, facecolor='green', alpha=0.5)
                f2_ax2.fill_between(self.Rmr_Perc, self.YSect_100, self.Rmr_Ht, where=self.Rmr_Ht < self.YSect_100, facecolor='red', alpha=0.5)
                f2_ax2.fill_between([0,self.Mr1+0.1], self.YSect_0, self.YSect_100, facecolor='blue', alpha=0.5)
                f2_ax2.fill_between(self.Rmr_Perc[:self.Rmr_Perc.index(self.Mr1)-len(self.Rmr_Perc)], self.YSect_100, self.Rmr_Ht[:self.Rmr_Perc.index(self.Mr1)-len(self.Rmr_Perc)], where=self.Rmr_Ht[:self.Rmr_Perc.index(self.Mr1)-len(self.Rmr_Perc)] > self.YSect_100, facecolor='blue', alpha=0.5)


                f2_ax2.plot([0,100],[self.YSect_0,self.YSect_0],color='black',marker="",ms=6,)
                f2_ax2.plot([0,100],[self.YSect_100,self.YSect_100],color='black',marker="",ms=6,)
                f2_ax2.set_facecolor(bgColor)



                if saveFig == True:
                        f2_ax1.plot(self.X_Array, np.full((640), self.crossX, dtype=int), 'b', linewidth=2., label="X Profile")
                        f2_ax1.plot(np.full((480), self.crossY, dtype=int), self.Y_Array, 'r', linewidth=2., label="Y Profile")
                        for txt in f2_ax1.texts:
                                txt.set_visible(False)

                        #Change label size to look better in output image
                        plotFontSize = 24
                        f2_ax1.set_title("Contour", fontsize=plotFontSize)
                        f2_ax1.set_xlabel("X", fontsize=plotFontSize)
                        f2_ax1.set_ylabel("Y", fontsize=plotFontSize)
                        f2_ax3.set_title("X Profile", fontsize=plotFontSize)
                        f2_ax3.set_xlabel("X", fontsize=plotFontSize)
                        f2_ax3.set_ylabel("Z", fontsize=plotFontSize)
                        f2_ax4.set_title("Y Profile", fontsize=plotFontSize)
                        f2_ax4.set_xlabel("Y", fontsize=plotFontSize)
                        f2_ax4.set_ylabel("Z", fontsize=plotFontSize)
                        f2_ax2.set_title('Bearing Ratio', fontsize=plotFontSize)
                        f2_ax2.set_ylabel('Height', fontsize=24)
                        f2_ax2.set_xlabel('Percent Area', fontsize=plotFontSize)



                        
                        SavePlot = self.FPath + self.NFName + "_Image.png"
                        plt.suptitle(self.PlotTitle, fontsize=32)
                        fig2.set_size_inches(32,18)
                        plt.savefig(SavePlot, bbox_inches='tight', facecolor=bgColor)

                        #Save the Data Results
                        self.saveRoughness()

                else:
                        plt.suptitle(self.PlotTitle, fontsize=20)
                        return fig2




        #Gets the updated profiles based on the crosshair clicks#########################################################################################################################################
        def getUpdatedProfile(self, xVal, yVal):
                self.crossX = xVal
                self.crossY = yVal
                return [self.X_Array, self.ModArr[int(float(xVal)/self.Meas_Pix)], self.Y_Array, self.ModArr[:,int(float(yVal)/self.Meas_Pix)]]





        def saveRoughness(self):
                resultsHead = [['DATE_TIMESTAMP','IDENTIFIER','IDENTIFIER_TYPE','ORIENTATION','UNIT_X','UNIT_Y','X_NOMINAL','Y_NOMINAL','FEATURE','PARAMETER','FEATURE_NUM','RESPONSE','DATA_TYPE','PF_FLAG','TESTGROUP','TESTDISPO','STRUCTURE']]
                

                resultsList = [[self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Ra','1',str("{:.6f}".format(float(round((self.Ra*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rp','1',str("{:.6f}".format(float(round((self.Rp*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rv','1',str("{:.6f}".format(float(round((self.Rv*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rt','1',str("{:.6f}".format(float(round((self.Rt*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rku','1',str("{:.6f}".format(float(round((self.Rku).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rq','1',str("{:.6f}".format(float(round((self.Rq*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rsk','1',str("{:.6f}".format(float(round((self.Rsk).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Mr1','1',str("{:.6f}".format(float(round((self.Mr1).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Mr2','1',str("{:.6f}".format(float(round((self.Mr2).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rk','1',str("{:.6f}".format(float(round((self.Rk*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rpk','1',str("{:.6f}".format(float(round((self.Rpk*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rvk','1',str("{:.6f}".format(float(round((self.Rvk*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-V1','1',str("{:.6f}".format(float(round((self.V1*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-V2','1',str("{:.6f}".format(float(round((self.V2*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rpm','1',str("{:.6f}".format(float(round((self.Rpm*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rvm','1',str("{:.6f}".format(float(round((self.Rvm*1000).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Surf','PRA-Rz','1',str("{:.6f}".format(float(round((self.Rz*1000).real,10)))),'Float','',self.StageX_Y,'','']]


                if os.path.exists(self.FPath+ self.OutputName):
                    SaveName = self.FPath + self.OutputName
                    file = open(SaveName, 'a')
                    writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_NONE, lineterminator='\n')
                    writer.writerows(resultsList)
                    #print SaveName
                    file.close()

                else:
                    SaveName = self.FPath + self.OutputName
                    file = open(SaveName, 'w')
                    writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_NONE, lineterminator='\n')
                    writer.writerows(resultsHead)
                    writer.writerows(resultsList)
                    #print SaveName
                    file.close()



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


        #Panel ID Binary############################################################################################################################################################################################
        def IDBinary(self, PanelID):
                BASE36 = str(PanelID)
                DECIMAL = int(BASE36,36)
                BINARY = str(bin(DECIMAL).split('b')[1]).zfill(24)
                return BINARY
            

        #Overwrite NaNs with column value interpolations################################################################################################################################################
        def interpNaN(self, NArray):
            for j in range(NArray.shape[1]):
                    mask_j = np.isnan(NArray[:,j])
                    NArray[mask_j,j] = np.interp(np.flatnonzero(mask_j), np.flatnonzero(~mask_j), NArray[~mask_j,j])
            return NArray

