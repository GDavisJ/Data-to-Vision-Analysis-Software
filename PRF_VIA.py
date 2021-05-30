"""
Module Name: PRF_VIA
Project: Data-to-Vision
Owner: Gary Davis
Class Description: This class is used to analyze via data.
        A LMS circular fit will be completed on the via top and bottom.
        The diameters and depth will be reported.
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
from skimage.feature import blob_dog, blob_doh, canny


class PRF_VIA(object):
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
                self.topCircle = []
                self.botCircle = []
                self.topDiam = 0.0
                self.botDiam = 0.0
                self.viaDepth = 0.0
                self.Offset = 0.0


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
                self.viaAnalysis()





        #Process the data and complete the via analysis
        def viaAnalysis(self):
                #Define the Top Threshold and Average top surface
                self.ModArr = self.interpNaN(self.ModArr)
                TempTop = self.ModArr
                TopHist = np.histogram(self.ModArr, bins=800)
                yhat_x = np.array(TopHist[1]).reshape(np.array(TopHist[1]).size).tolist()
                yhat = PRF_Algo().savitzky_golay(np.array(np.array(TopHist[0]).reshape(np.array(TopHist[0]).size).tolist()).astype('float'), 25, 6)
                MaximaListBins = np.array(yhat)[argrelextrema(np.array(yhat), np.greater,order=1)[0]]
                MaximaListHeight = np.array(yhat_x)[argrelextrema(np.array(yhat), np.greater,order=1)[0]]
                MaximaVal = float(np.array(MaximaListHeight).reshape(np.array(MaximaListHeight).size).tolist()[np.array(MaximaListBins).reshape(np.array(MaximaListBins).size).tolist().index(np.max(MaximaListBins))])
                MinimaListBins = np.array(yhat)[argrelextrema(np.array(yhat), np.less,order=20)[0]]
                MinimaListHeight = np.array(yhat_x)[argrelextrema(np.array(yhat), np.less,order=20)[0]]
                MinimaListHeight1 = np.array(MinimaListHeight).reshape(np.array(MinimaListHeight).size).tolist()
                MinimaListHeight1.reverse()
                FoundMin = False
                NewThresh = 0.
                for CutoffMinima in MinimaListHeight1:
                    if float(CutoffMinima) <= float(MaximaVal) and FoundMin == False and float(CutoffMinima) < 100:
                        FoundMin = True
                        NewThresh = float(CutoffMinima)
                        NewThreshHt =  MinimaListHeight1[MinimaListHeight1.index(CutoffMinima)+1]
##                        print NewThresh



                TopHeight = self.SortAndFill(self.ModArr, NewThresh, np.nan, 'LE')
                Heightmask =~np.isnan(TopHeight)
                AvgTopHt = np.average(TopHeight[Heightmask])

                
                XV1, YV1 = np.meshgrid(self.X_Array,self.Y_Array)
                XV3, YV3 = np.meshgrid(self.X_Array,self.Y_Array)


                TempTest = self.SortAndFill(self.ModArr, NewThresh, 0, 'GE')  
                TempTest = self.SortAndFill(TempTest, 0, 255, 'L')
                
                #use blob detection to find the via locations
                Blob = blob_dog(TempTest,min_sigma=10,max_sigma=150., exclude_border=True)
                Blob[:, 2] = Blob[:, 2] * np.sqrt(2)
                #y, x, r = Blob

                #Find the via closest to the center of FOV.
                Img_X_Cent = 320
                Img_Y_Cent = 240
                Near_BlobX = 0
                Near_BlobY = 0
                OVal = 0
                Near_R = 0
                Offset_Cent = np.sqrt(np.square(Img_X_Cent) + np.square(Img_X_Cent))
                for CentBlob in Blob:
                    Offset = np.sqrt(np.square(Img_X_Cent - CentBlob[1]) + np.square(Img_Y_Cent - CentBlob[0]))
                    Off_Dif = np.abs(Offset_Cent - Offset)
                    if float(Off_Dif) >= float(OVal) and CentBlob[2] > (5* np.sqrt(2)):
                        OVal = Off_Dif
                        Near_BlobX = CentBlob[1]
                        Near_BlobY = CentBlob[0]
                        Near_R = CentBlob[2]
                        

##                #Closest to the center of FOV.                
##                centerblob = [(Near_BlobX*self.Meas_Pix) , (Near_BlobY*self.Meas_Pix) ]

                #Get indices for bounding mask (L=leftEdge, R=rightEdge, T=topEdge, B=bottomEdge)
                R = int((Near_BlobX + 2) + (np.round(Near_R)+10))
                L = int((Near_BlobX - 2) - (np.round(Near_R)+10))
                T = int((Near_BlobY + 2) + (np.round(Near_R)+10))
                B = int((Near_BlobY - 2) - (np.round(Near_R)+10))

                #create a temp array filled with zero, add selected data and binarize it.
                Segmented = np.zeros(shape=(480,640))
                Segmented[B:T,L:R] = self.ModArr[B:T,L:R]
                Segmented = self.SortAndFill(Segmented, NewThresh, 0, 'G')        
                Segmented = self.SortAndFill(Segmented, 0, 255, 'L')

                #Use edge detection
                Edge = canny(Segmented, sigma=9, low_threshold = 1)
                mask =~np.isnan(Edge)
                XV1 = np.array(XV1[Edge])
                YV1 = np.array(YV1[Edge])

                #LMS fit of top via data
                self.topCircle = PRF_Algo().leastsq_circle(XV1,YV1)
                self.topDiam = self.topCircle[9]
##                print("Circle Fit Data: ", self.topCircle)
                print ("Top Diameter: ", self.topCircle[9])


                #Prep via bottom for analysis
                TempBot = np.empty(shape=(480,640))
                TempBot[:] = np.nan
                TempBot[B:T,L:R] = self.ModArr[B:T,L:R]
                
                TempBot = self.SortAndFill(TempBot, NewThreshHt, np.nan, 'G')
                BotMask =~np.isnan(TempBot)
                BotHist = np.histogram(TempBot[BotMask], bins=800)

                #A histogram is used to find the frequency distribution. The signal is noisy so a smoothing algorithm is used to find transitions.
                yhat_x = np.array(BotHist[1]).reshape(np.array(BotHist[1]).size).tolist()
                yhat = PRF_Algo().savitzky_golay(np.array(np.array(BotHist[0]).reshape(np.array(BotHist[0]).size).tolist()).astype('float'), 25, 6) # window size 51, polynomial order 3
                MaximaListBins = np.array(yhat)[argrelextrema(np.array(yhat), np.greater,order=6)[0]]
                MaximaListHeight = np.array(yhat_x)[argrelextrema(np.array(yhat), np.greater,order=6)[0]]
                MaximaVal = float(np.array(MaximaListHeight).reshape(np.array(MaximaListHeight).size).tolist()[np.array(MaximaListBins).reshape(np.array(MaximaListBins).size).tolist().index(np.max(MaximaListBins))])
                MinimaListBins = np.array(yhat)[argrelextrema(np.array(yhat), np.less,order=8)[0]] #Order was 12 and 8
                MinimaListHeight = np.array(yhat_x)[argrelextrema(np.array(yhat), np.less,order=8)[0]]

                maxMid = np.max(MaximaListBins)*0.5 #This uses 50% of the max bin height as a threshold
                FoundMin = False
                NewThreshHt = NewThreshHt
                CntLst = 0
                CntLstVal = 0
                for CutoffMinima in MinimaListHeight:
                    CntLst = CntLst + 1
                    MinimaBinVal = float(np.array(MinimaListBins).reshape(np.array(MinimaListBins).size).tolist()[np.array(MinimaListHeight).reshape(np.array(MinimaListHeight).size).tolist().index(CutoffMinima)])
                    if float(CutoffMinima) >= float(MaximaVal) and FoundMin == False and MinimaBinVal < maxMid:
                        FoundMin = True
                        NewThreshHt = float(CutoffMinima)
                        MinimaBinVal = float(np.array(MinimaListBins).reshape(np.array(MinimaListBins).size).tolist()[np.array(MinimaListHeight).reshape(np.array(MinimaListHeight).size).tolist().index(CutoffMinima)])
                        CntLstVal = CntLst



                Segmented = np.zeros(shape=(480,640))
                Segmented[B:T,L:R] = self.ModArr[B:T,L:R]

                BotHeight = self.SortAndFill(Segmented, NewThreshHt, np.nan, 'G')
                Heightmask =~np.isnan(BotHeight)
                AvgBotHt = np.average(BotHeight[Heightmask])
##                print "Avg Bot Ht: " + str(AvgBotHt)
##                print "Via Depth: " + str(AvgTopHt - AvgBotHt)
                self.viaDepth = AvgTopHt - AvgBotHt


                Segmented = self.SortAndFill(Segmented, NewThreshHt, 0, 'G')        
                Segmented = self.SortAndFill(Segmented, 0, 255, 'L')

                
                tmpSigma = .25
                fillComplete = False
                while fillComplete == False:
                        test = canny(Segmented, sigma=tmpSigma)
                        test = ndimage.binary_closing(test,iterations=5)
                        test = ndimage.binary_fill_holes(test)
                        test = ndimage.binary_erosion(test, iterations=5)

                        #This logic is used to make sure that the via bottom is filled so false data isn't used.
                        if test[int(Near_BlobY)][int(Near_BlobX)] !=0:
                            Edge = canny(test)
                            mask =~np.isnan(Edge)
                            XA1 = np.array(XV3[Edge])
                            YA1 = np.array(YV3[Edge])
                            try:
                                self.botCircle = PRF_Algo().leastsq_circle(XA1,YA1)
                                self.botDiam = self.botCircle[9]
                                print ("Bottom Diameter: ", self.botCircle[9])
                                fillComplete = True
                            except:
                                10/0 #Force fail for now
                            if np.count_nonzero(Edge == True) < 150:
                                Edge = canny(Segmented, sigma=2, low_threshold = 30)

                        else:
                            tmpSigma = tmpSigma + 0.25
                            if tmpSigma >= 5: #needed or will be in infinate loop!!!
                                fillComplete = True

                self.Offset = np.sqrt(np.square(float(self.topCircle[0])-float(self.botCircle[0]))+np.square(float(self.topCircle[1])-float(self.botCircle[1])))


                
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
                f2_ax1.plot (self.topCircle[6], self.topCircle[7], 'b-', linewidth=2.0, label="Top Circle Fit") #Circle Fit plot
                f2_ax1.plot (self.botCircle[6], self.botCircle[7], 'r-', linewidth=2.0, label="Bot Circle Fit") #Circle Fit plot
                contourf_ = f2_ax1.contourf(self.X_Array, self.Y_Array, self.ModArr , 60, cmap=plt.get_cmap('jet'))#, cmap=plt.cm.mpl)
                fig2.colorbar(contourf_, format='%.4f', ax=f2_ax1)
                f2_ax1.set_title("Contour", fontsize=plotFontSize)
                f2_ax1.set_xlabel("X", fontsize=plotFontSize)
                f2_ax1.set_ylabel("Y", fontsize=plotFontSize)
                txtPos = self.X_Array[-1]*-1.16239
                f2_ax1.text(txtPos, int(479*self.Meas_Pix), 'Total Mag:    ' + str(round(self.Mag_Val,1))+"x",fontsize=12)
                f2_ax1.text(txtPos, int(420*self.Meas_Pix), 'Pixel Size:   ' + str(round(self.Meas_Pix,4))+"um",fontsize=12)
                f2_ax1.text(txtPos, int(340*self.Meas_Pix), 'Top Diam:   ' + str(round(self.topDiam,4))+"um",fontsize=12)
                f2_ax1.text(txtPos, int(280*self.Meas_Pix), 'Bot Diam:   ' + str(round(self.botDiam,4))+"um",fontsize=12)
                f2_ax1.text(txtPos, int(220*self.Meas_Pix), 'Depth:        ' + str(round(self.viaDepth,4))+"um",fontsize=12)
##                f2_ax1.text(txtPos, int(160*self.Meas_Pix), 'Rsk:        ' + str(round(self.Rsk,4)),fontsize=12)


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
                        f2_ax2.set_ylabel('Pts', fontsize=24)
                        f2_ax2.set_xlabel('Height', fontsize=plotFontSize)



                        
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
                

                resultsList = [[self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Via-Top','PVA-CF-Diameter','1',str("{:.6f}".format(float(round((self.topDiam).real,3)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Via-Bot','PVA-CF-Diameter','1',str("{:.6f}".format(float(round((self.botDiam).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Via','PVA-Depth','1',str("{:.6f}".format(float(round((self.viaDepth).real,10)))),'Float','',self.StageX_Y,'',''],
                               [self.DateTimeStamp,self.BINARY,'PANEL_ID',self.PSide,self.Unit_X,self.Unit_Y,self.Xnom,self.Ynom,'Via','PVA-Offset','1',str("{:.6f}".format(float(round((self.Offset).real,10)))),'Float','',self.StageX_Y,'','']]




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

