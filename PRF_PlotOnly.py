"""
Module Name: PRF_PlotOnly
Project: Data-to-Vision
Owner: Gary Davis
Class Description: This class is used to create plots for the dataset.
        These plots will be returned to the view and displayed on the GUI.
"""

import  csv, datetime, time
import numpy as np
import numpy.ma as ma
from PRF_Algorithms import PRF_Algo
from matplotlib import pyplot as plt, cm, colors
import scipy.ndimage as ndimage
import matplotlib.gridspec as gridspec


class PRF_Plot(object):
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


                
        #Overwrite NaNs with column value interpolations################################################################################################################################################
        def interpNaN(self, NArray):
            for j in range(NArray.shape[1]):
                    mask_j = np.isnan(NArray[:,j])
                    NArray[mask_j,j] = np.interp(np.flatnonzero(mask_j), np.flatnonzero(~mask_j), NArray[~mask_j,j])
            return NArray


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
                #f2_ax2 = fig2.add_subplot(spec2[0, 1], projection='3d')
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



                #Histogram Plot
                mask =~np.isnan(self.ModArr)
                f2_ax2.set_title('Histogram', fontsize=plotFontSize)
                f2_ax2.set_ylabel('Pts', fontsize=plotFontSize)
                f2_ax2.set_xlabel('Height', fontsize=plotFontSize)
                counts, bins = np.histogram(self.ModArr[mask], bins=200)
                f2_ax2.hist(bins[:-1], bins, weights=counts)
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
                        f2_ax2.set_title('Histogram', fontsize=plotFontSize)
                        f2_ax2.set_ylabel('Pts', fontsize=plotFontSize)
                        f2_ax2.set_xlabel('Height', fontsize=plotFontSize)

                        
                        SavePlot = self.FPath + self.NFName + "_Image.png"
                        plt.suptitle(self.PlotTitle, fontsize=32)
                        fig2.set_size_inches(32,18)
                        plt.savefig(SavePlot, bbox_inches='tight', facecolor=bgColor)

                else:
                        plt.suptitle(self.PlotTitle, fontsize=20)
                        return fig2



        #Gets the updated profiles based on the crosshair clicks#########################################################################################################################################
        def getUpdatedProfile(self, xVal, yVal):
                self.crossX = xVal
                self.crossY = yVal
                return [self.X_Array, self.ModArr[int(float(xVal)/self.Meas_Pix)], self.Y_Array, self.ModArr[:,int(float(yVal)/self.Meas_Pix)]]


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

