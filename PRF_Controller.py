"""
Module Name: PRF_Controller
Project: Data-to-Vision
Owner: Gary Davis
Class Description: This class is the controller and it passes information/data between the
    Model and View.
"""

from PRF_ROUGH import PRF_ROUGHNESS
from PRF_PlotOnly import PRF_Plot
from PRF_VIA import PRF_VIA
from PRF_PAD import PRF_PAD

class PRF_Controller(object):
    #Constructor used to create the correct object based on the user inputs.
    def __init__(self, fname, fpath, analysis, tipTilt, filt):
        self.FName = fname
        self.FPath = fpath
        self.selectedAnalysis = analysis
        self.tipTilt = tipTilt
        self.selectedFilt = filt
        self.__analyObj = None

        if self.selectedAnalysis == 'Plot Only':
            self.analyObj = PRF_Plot(self.FName, self.FPath, self.tipTilt, self.selectedFilt)

        elif self.selectedAnalysis == 'Roughness':
            self.analyObj = PRF_ROUGHNESS(self.FName, self.FPath, self.tipTilt, self.selectedFilt)

        elif self.selectedAnalysis == 'Via':
            self.analyObj = PRF_VIA(self.FName, self.FPath, self.tipTilt, self.selectedFilt)

        elif self.selectedAnalysis == 'Pad':
            self.analyObj = PRF_PAD(self.FName, self.FPath, self.tipTilt, self.selectedFilt)

    #Method used to update the plot theme                        
    def getFigObj(self, bgColor='#FFFFFF', saveFig=False):
        return self.analyObj.getFigObj(bgColor, saveFig)

    #Method used to pass the x/y click event to the analysis object and return updated plots.
    def getProfiles(self, xval, yval):
        return self.analyObj.getUpdatedProfile(xval, yval)

    #Method used to update the object properties based on user input.
    def updateProperties(self, tipTilt, filt):
        self.tipTilt = tipTilt
        self.selectedFilt = filt
        self.analyObj.analysisChange(self.tipTilt, self.selectedFilt)
