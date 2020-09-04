from PRF_ROUGH import PRF_ROUGHNESS
from PRF_PlotOnly import PRF_Plot

class PRF_Controller(object):
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

    def getFigObj(self, bgColor='#FFFFFF', saveFig=False):
        return self.analyObj.getFigObj(bgColor, saveFig)

    def getProfiles(self, xval, yval):
        return self.analyObj.getUpdatedProfile(xval, yval)

    def updateProperties(self, tipTilt, filt):
        self.tipTilt = tipTilt
        self.selectedFilt = filt
        self.analyObj.analysisChange(self.tipTilt, self.selectedFilt)
