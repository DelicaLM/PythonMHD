#FileSaver.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Defines the FileSaver class for saving figures and/or data during PythonMHD simulations. If a user has set
#         saveFigs and/or saveData to true in their simulation parameters, PythonMHD will create a FileSaver object
#         when it creates the Simulation object for the user.

##########IMPORT STATEMENTS###########

#Import numpy for mathematical operations
import numpy as np

#Import os for navigating between file directories
import os

#Import datetime for retrieving the date and time when we create output files
from datetime import datetime

#Import SciPy module for creating .mat files
import scipy.io

#Import pyevtk library for creating .vtk files
from pyevtk.hl import pointsToVTK

#Import glob for retrieving the image files that need to be included in simulation movies
import glob

#Import cv2 for converting the image files into a .mp4 movie
import cv2

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants

###############FILE SAVER CLASS#################
#Class: FileSaver
#Purpose: A FileSaver object manages all of the file creation details (e.g., creating output folders) that are required
#         to save the figures and/or data for PythonMHD simulations.
class FileSaver:
    #########CONSTRUCTOR METHOD##########
    #Function: __init__
    #Purpose: Creates a new FileSaver object (if the user provides valid
    #         inputs for the primitive variables matrix and the grid parameters).
    #Inputs: saveFigs (boolean for whether the FileSaver will need to save figures/images)
    #        saveData (boolean for whether the FileSaver will need to save numerical data)
    #        saveFigsDt (amount of simulation time between visualization saves)
    #        saveDataDt (amount of simulation time between numerical data saves)
    #        outputFolderPathInput (the user's input for the path to the location on their computer where they
    #                               want PythonMHD to create their outputs folder) (empty string if the user has
    #                               not provided this parameter in simPar) (If the user does not provide an output folder
    #                               path, PythonMHD will save their output folder in PythonMHD's Outputs directory.)
    #                               (empty string if the user has not provided this parameter in simPar)
    #        outputFolderNameInput (the user's input for the name for the folder where all outputs should be saved)
    #                              (empty string if the user has not provided this parameter in simPar)
    #        outputFileNameInput (the user's input for the prefix name for all of the output files) (after this prefix,
    #                             PythonMHD appends the time in the simulation so that outputs for different timesteps
    #                             have different file names) (empty string if the user has not provided this parameter
    #                             in simPar)
    def __init__(self, saveFigs, saveData, tLim, saveFigDt=-1, saveDataDt=-1,
                 outputFolderPathInput="",outputFolderNameInput="", outputFileNameInput=""):
        #First we need to determine the path to where we should create the outputs folder
        self.outputFolderPath = ""
        #If the user has provided an outputFolderPath parameter, use that path.
        if outputFolderPathInput != "":
            #Make sure the user's requested path actually exists
            assert os.path.exists(outputFolderPathInput), "The outputFolderPath provided in simPar, " + outputFolderPathInput\
                                                          + " is not a valid path on your computer. Please provide a path "\
                                                          + "that exists or remove outputFolderPath from simPar. If you do "\
                                                          + "not provide an outputFolderPath parameter in simPar, PythonMHD "\
                                                          + "will save your output folder in its own Outputs director."
            self.outputFolderPath = outputFolderPathInput
        else: #if the user has not specified a path to the output folder
            #If the user does not provide an outputFolderPath parameter, we will put the output folder in PythonMHD's
            #Outputs directory.
            #Get the absolute path to this file (because this will tell us where PythonMHD is located on the user's machine)
            abspath = os.path.abspath((__file__))
            #Get path to the PythonMHD Source directory
            dname = os.path.dirname(abspath)
            #Set Source as the current directory
            os.chdir(dname)
            #Get the directory that Source belongs to (this is the main PythonMHD directory)
            pythonMHDDir = os.path.dirname(os.getcwd())
            #Add "Outputs" to the path
            self.outputFolderPath = pythonMHDDir + "/Outputs/"
            #Create the Outputs folder if it doesn't exist
            if not os.path.exists(self.outputFolderPath):
                os.mkdir(self.outputFolderPath)
        #Now we need to determine the name for the output folder
        self.outputFolderName = ""
        #Check if the user has provided a name for their output folder
        if outputFolderNameInput != "":
            self.outputFolderName = outputFolderNameInput
        else: #if the user has not provided an outputFolderName paramter in simPar
            #The default output folder name includes the date/time so the user can
            #more easily identify the simulation.
            currDateTime = datetime.now()
            self.outputFolderName = "PythonMHD_Simulation_Date_" + currDateTime.strftime("month%m_day%d_year%Y_") \
                                    + "Time_" + currDateTime.strftime("%Hoclock%Mmin") + "_"
        #Make sure that at least one of saveFigs and saveData is true (if we aren't
        #saving anything, PythonMHD shouldn't be creating a FileSaver object)
        assert saveFigs or saveData, "FileSaver Error: A FileSaver object was created even though the simulation "\
                                    + "does not need to save any data."
        self.saveFigs = saveFigs
        self.saveData = saveData
        #Now we will create the outputs folder where we will save all images and/or data
        self.fullPath = self.outputFolderPath + "/" + self.outputFolderName + "/"
        #Check if there is already a folder with the same name
        #(so we don't accidentally overwrite data for another simulation)
        if os.path.exists(self.fullPath):
            #We will create a unique folder name by appending a number
            #Keep increasing the number by 1 until we find a unique folder name
            i = 1
            newFolder = self.outputFolderName + "_" + str(i)
            self.fullPath = self.outputFolderPath + "/" + newFolder + "/"
            while os.path.exists(self.fullPath):
                i += 1
                newFolder = self.outputFolderName + "_" + str(i)
                self.fullPath = self.outputFolderPath + "/" + newFolder + "/"
            #Save the new output folder name
            self.outputFolderName = newFolder
        #Create the output folder
        os.mkdir(self.fullPath)
        #Create a sub-folder for figures, if we are saving visualizations
        if self.saveFigs:
            os.mkdir(self.fullPath + "/Figures/")
        #Create a sub-folder for .mat files, if we are saving numerical data
        if self.saveData:
            os.mkdir(self.fullPath + "/Data/")
        #Next we need to determine the prefix string for all of our output file name
        self.outputFileName = ""
        #Use the file name passed by the user if they provided one
        if outputFileNameInput != "":
            self.outputFileName = outputFileNameInput
        else:
            #If the user has not provided a file name, use the output folder name
            self.outputFileName = self.outputFolderName
        #Get the amount of simulation time between visualization saves (-1 if we are not saving visualizations)
        self.saveFigDt = saveFigDt
        #Get the amount of simulation time between data saves (-1 if we are not saving numerical data)
        self.saveDataDt = saveDataDt
        #Figure out how many decimal places we need to represent the time limit for the simulation
        #(useful for when we need to append the time limit to the file name for the last figure/data file)
        tLimPrec = 1
        while np.abs(np.power(10, tLimPrec)*tLim - int(np.power(10, tLimPrec)*tLim)) > constants.TIME_PRECISION_SMALL_VAL \
                and tLimPrec < constants.MAX_FILE_NAME_TIME_PREC:
            tLimPrec += 1
        #Figure out how many decimal places we should include when we append the simulation time
        #to output file names for visualizations (if we are saving matplotlib figures)
        self.figTimePrec = -1
        if self.saveFigs:
            self.figTimePrec = 1
            while np.abs(np.power(10,self.figTimePrec)*self.saveFigDt
                         - int(np.power(10,self.figTimePrec)*self.saveFigDt)) > constants.TIME_PRECISION_SMALL_VAL\
                  and self.figTimePrec < constants.MAX_FILE_NAME_TIME_PREC:
                self.figTimePrec += 1
            #Use the time limit precision if it requires more decimal places than saveFigDt
            self.figTimePrec = np.maximum(tLimPrec,self.figTimePrec)

        #Figure out how many decimal places we should include when we append the simulation time
        #to output file names for numerical data (if we are saving numerical data)
        self.dataTimePrec = -1
        if self.saveData:
            self.dataTimePrec = 1
            while np.abs(np.power(10,self.dataTimePrec)*self.saveDataDt
                         - int(np.power(10,self.dataTimePrec)*self.saveDataDt)) > constants.TIME_PRECISION_SMALL_VAL\
            and self.dataTimePrec < constants.MAX_FILE_NAME_TIME_PREC:
                self.dataTimePrec += 1
            #Use the time limit precision if it requires more decimal places than saveDataDt
            self.dataTimePrec = np.maximum(tLimPrec, self.dataTimePrec)


    ######PUBLIC METHODS#######
    #Function: saveFigures
    #Purpose: Saves all of the figures in a SimVisualizer object as image files (e.g., "png", "pdf", etc.).
    #Inputs: visualizer (the visualizer object that contains the matplotlib figures to save)
    #        simTime (the time in the simulation (for including the time in file names))
    #Outputs: None (but all of the files that this function creates will now be available
    #               in the user's outputs folder)
    def saveFigures(self, visualizer, simTime):
        print("Saving Figures")
        #Iterate over the matplotlib figures
        for figNum in range(visualizer.numFigures):
            #Get the parameters for the current figure
            figPar = visualizer.figures[figNum]
            #Check if we are supposed to save this particular figure
            saveFig = figPar[constants.SAVE_FIGS]
            if saveFig:
                #Get the figure object
                currFigure = figPar[constants.FIGURE]
                #Get the figure title
                figureTitle = figPar[constants.FIGURE_TITLE].replace(" ","_")
                #Get the output file type
                figureFormat = figPar[constants.SAVE_FIGS_FORMAT]
                #Get the output resolution in dpi/dots-per-inch
                figureDPI = figPar[constants.SAVE_FIGS_DPI]
                #Check if we have a folder already for this figure (make one if we don't)
                figureFolderPath = self.fullPath + "/Figures/" + figureTitle + "/"
                if not os.path.exists(figureFolderPath):
                    os.mkdir(figureFolderPath)
                #Save the figure
                currFigure.savefig(figureFolderPath + self.outputFileName + "_" + figureTitle + "_Output_time_"
                                   + ("{:." + str(self.figTimePrec) + "f}").format(simTime) + "." + figureFormat,
                                   format=figureFormat,
                                   dpi=figureDPI)

    #Function: saveData
    #Purpose: Saves all of the numerical data in a simulation grid as .mat or .vtk files
    #Inputs: simGrid (the SimulationGrid object that contains all of the data we need to save)
    #        simTime (the time in the simulation (for including the time in file names))
    #        dataFormat (the format in which to save the output data (.mat or .vtk))
    #Outputs: None (but all of the files that this function creates will now be available
    #               in the user's outputs folder)
    def saveDataFiles(self, simGrid, simTime, dataFormat):
        print("Saving Data")
        if dataFormat == constants.DATA_FORMAT_MAT:
            if not simGrid.isMHD:
                scipy.io.savemat(self.fullPath + "/Data/" + self.outputFileName
                                 + ("_Output_Time_{:." + str(self.dataTimePrec) + "f}").format(simTime) + ".mat",
                                 {constants.DENSITY: simGrid.primVars[0], constants.VX: simGrid.primVars[1],
                                  constants.VY: simGrid.primVars[2], constants.VZ: simGrid.primVars[3],
                                  constants.PRESSURE: simGrid.primVars[4], constants.ENERGY: simGrid.consVars[4]})
            else:
                scipy.io.savemat(self.fullPath + "/Data/" + self.outputFileName
                                 + ("_Output_Time_{:." + str(self.dataTimePrec) + "f}").format(simTime) + ".mat",
                                 {constants.DENSITY: simGrid.primVars[0], constants.VX: simGrid.primVars[1],
                                  constants.VY: simGrid.primVars[2], constants.VZ: simGrid.primVars[3],
                                  constants.BX: simGrid.primVars[4], constants.BY: simGrid.primVars[5],
                                  constants.BZ: simGrid.primVars[6], constants.PRESSURE: simGrid.primVars[7],
                                  constants.ENERGY: simGrid.consVars[7]})
        else:
            xPoints = simGrid.xCoords.flatten()
            yPoints = np.zeros(shape=xPoints.shape)
            zPoints = np.zeros(shape=xPoints.shape)
            if simGrid.nDim >= 2:
                yPoints = simGrid.yCoords.flatten()
            elif simGrid.nDim == 3:
                zPoints = np.flatten(simGrid.zCoords)
            if not simGrid.isMHD:
                pointsToVTK(self.fullPath + "/Data/" + self.outputFileName
                            + ("_Output_Time_{:." + str(self.dataTimePrec) + "f}").format(simTime),
                            xPoints, yPoints, zPoints,
                            data={constants.DENSITY:simGrid.primVars[0].flatten(),
                                  constants.VX:simGrid.primVars[1].flatten(),
                                  constants.VY:simGrid.primVars[2].flatten(),
                                  constants.VZ:simGrid.primVars[3].flatten(),
                                  constants.PRESSURE:simGrid.primVars[4].flatten(),
                                  constants.ENERGY:simGrid.consVars[4].flatten()})
            else:
                pointsToVTK(self.fullPath + "/Data/" + self.outputFileName
                            + ("_Output_Time_{:." + str(self.dataTimePrec) + "f}").format(simTime),
                            xPoints, yPoints, zPoints,
                            data={constants.DENSITY: np.flatten(simGrid.primVars[0]),
                                  constants.VX: simGrid.primVars[1].flatten(),
                                  constants.VY: simGrid.primVars[2].flatten(),
                                  constants.VZ: simGrid.primVars[3].flatten(),
                                  constants.BX: simGrid.primVars[4].flatten(),
                                  constants.BY: simGrid.primVars[5].flatten(),
                                  constants.BZ: simGrid.primVars[6].flatten(),
                                  constants.PRESSURE: simGrid.primVars[7].flatten(),
                                  constants.ENERGY: simGrid.consVars[7].flatten()})

    #Function: makeMovies
    #Purpose: Makes a .mp4 movie of every matplotlib figure with saved images for the simulation.
    #Inputs: visualizer (the visualizer object that contains the matplotlib figures)
    #        frameRate (the frame rate)
    #Outputs: None (but all of the files that this function creates will now be available
    #               in the user's outputs folder)
    def makeMovies(self, visualizer):
        print("Saving Figures")
        #Iterate over the matplotlib figures
        for figNum in range(visualizer.numFigures):
            #Get the parameters for the current figure
            figPar = visualizer.figures[figNum]
            #Check if we are supposed to save this particular figure
            saveFig = figPar[constants.SAVE_FIGS]
            if saveFig:
                #Get the figure object
                currFigure = figPar[constants.FIGURE]
                #Get the figure title
                figureTitle = figPar[constants.FIGURE_TITLE].replace(" ","_")
                #Get the output file type
                figureFormat = figPar[constants.SAVE_FIGS_FORMAT]
                #Check if we have a folder of saved images for this figure
                figureFolderPath = self.fullPath + "/Figures/" + figureTitle + "/"
                if os.path.exists(figureFolderPath):
                    if figPar[constants.MAKE_MOVIE]:
                        images = []
                        for imageFile in sorted(glob.glob(figureFolderPath+ "*." + figureFormat)):
                            nextImage = cv2.imread(imageFile)
                            images.append(nextImage)
                        if len(images) > 0:
                            size = (images[0].shape[1], images[0].shape[0])
                            frameRate = constants.DEFAULT_FRAME_RATE
                            videoWriter = cv2.VideoWriter(figureFolderPath + self.outputFileName + ".mp4",
                                                          cv2.VideoWriter_fourcc(*'mp4v'), frameRate, size)
                            for i in range(len(images)):
                                videoWriter.write(images[i])
                            videoWriter.release()


