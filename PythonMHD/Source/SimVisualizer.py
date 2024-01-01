#SimVisualizer.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Defines the SimVisualizer class, which is used to manage PythonMHD visualizations.

######IMPORT STATEMENTS######

#Import matplotlib for creating figures and data plots
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import mplot3d
#Colorbar animation: https://www.tutorialspoint.com/how-to-animate-the-colorbar-in-matplotlib

#Import NumPy for working with matrices (necessary for checking whether the user's
#visPar parameters are consistent with their simulation grid)
import numpy as np

#Import the helper module for validating the user's inputs for visPar
import Source.SimVisualizer_HelperModule as simVisLib

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants


#######SIM VISUALIZER CLASS#######
class SimVisualizer:
    #######CONSTRUCTOR METHOD#######
    #Function: __init__
    #Purpose: Creates a new SimVisualizer object (if the user's visualization parameters (visPar)
    #         are valid and consistent with their simulation (e.g., whether it is MHD or hydro, the
    #         number of spatial directions, etc.)).
    #Inputs: simulationGrid (the Simulation Grid object that contains the initial data for the simulation)
    #        visPar (the visualization parameters)
    def __init__(self, simulationGrid, visPar, plotDt, tLim,
                 matplotlibBackend=constants.DEFAULT_MATPLOTLIB_BACKEND,
                 defaultSaveFigs=constants.DEFAULT_SAVE_FIGS,
                 defaultSaveFigFormat=constants.DEFAULT_IMAGE_FORMAT,
                 defaultSaveFigDPI=constants.DEFAULT_SAVE_FIGS_DPI,
                 defaultMakeMovie=constants.DEFAULT_MAKE_MOVIE):
        #Ensure that the simulation grid is 1D, 2D or 3D
        assert (simulationGrid.nDim == 1 or simulationGrid.nDim == 2 or simulationGrid.nDim == 3)
        #Use the SimVisualizer helper module to validate the user's figures and plots in visPar
        simVisLib.validateVisPar(simulationGrid,visPar,defaultSaveFigs)
        #Set the matplotlib backend
        print("Setting matplotlib backend \n(If your code fails soon after this step "
              +"you probably\n don't have the necessary libraries installed for the backend you chose.)")
        matplotlib.use(matplotlibBackend)
        #Determine how many figures the user wants to create
        self.numFigures = len(visPar[constants.FIGURES])
        #Ensure they want to create at least one figure
        assert self.numFigures >= 1
        #Create an empty array for the figures
        self.figures = [None]*self.numFigures

        #Figure out how many decimal places we will need to display the simulation time
        #in plot figures
        self.timePrec = 1
        while np.abs(np.power(10,self.timePrec)*plotDt
                     - int(np.power(10,self.timePrec)*plotDt)) > constants.TIME_PRECISION_SMALL_VAL\
        and self.timePrec < constants.MAX_PLOT_TITLE_TIME_PREC:
            self.timePrec += 1
        tLimPrec = 1
        while np.abs(np.power(10,tLimPrec)*tLim - int(np.power(10,tLimPrec)*tLim)) > constants.TIME_PRECISION_SMALL_VAL\
        and tLimPrec < constants.MAX_PLOT_TITLE_TIME_PREC:
            tLimPrec += 1
        self.timePrec = np.maximum(self.timePrec, tLimPrec)

        #For any figures that we want to save, we will need to make sure that
        #the user has requested a valid output image format (e.g., "png", "pdf", etc.)
        #for their matplotlib backend.
        imageFormatsToCheck = []

        #Retrieve parameter strings from the PythonMHD constants file
        #that will help us read values from visPar
        figuresString = constants.FIGURES
        figString = constants.FIGURE
        figTitleString = constants.FIGURE_TITLE
        saveFigsString = constants.SAVE_FIGS
        makeMovieString = constants.MAKE_MOVIE
        figFormatString = constants.SAVE_FIGS_FORMAT
        figDPIString = constants.SAVE_FIGS_DPI
        plotsString = constants.PLOTS
        colorbarsString = constants.COLORBARS
        axesString = constants.AXES
        numPlotsString = constants.NUM_PLOTS
        plotVarString = constants.PLOT_VAR
        plotTypeString = constants.PLOT_TYPE
        plotColorString = constants.PLOT_COLOR
        plotAlphaString = constants.PLOT_ALPHA
        minPlotValString = constants.MIN_PLOT_VAL
        hasMinPlotValString = constants.HAS_MIN_PLOT_VAL
        maxPlotValString = constants.MAX_PLOT_VAL
        hasMaxPlotValString = constants.HAS_MAX_PLOT_VAL
        plotAxisString = constants.PLOT_AXIS
        plotPlaneString = constants.PLOT_PLANE
        xPosString = constants.PLOT_X_POS
        yPosString = constants.PLOT_Y_POS
        zPosString = constants.PLOT_Z_POS
        xPosPrecString = constants.PLOT_X_POS_PREC
        yPosPrecString = constants.PLOT_Y_POS_PREC
        zPosPrecString = constants.PLOT_Z_POS_PREC
        numRowsString = constants.FIGURE_NUM_ROWS
        numColsString = constants.FIGURE_NUM_COLS
        #Set the default plot type and color based on
        #whether the simulation is 1D, 2D, or 3D
        defaultPlotType = constants.PLOT_TYPE_1D
        defaultPlotColor = constants.DEFAULT_PLOT_COLOR_1D
        if simulationGrid.nDim == 2:
            defaultPlotType = constants.PLOT_TYPE_2D
            defaultPlotColor = constants.DEFAULT_COLORMAP_2D
        elif simulationGrid.nDim == 3:
            defaultPlotType = constants.PLOT_TYPE_3D
            defaultPlotColor = constants.DEFAULT_COLORMAP_3D
        #Iterate over the figures in visPar
        for figNum in range(self.numFigures):
            #Get the parameters for the current figure
            figPar = visPar[figuresString][figNum]
            #Get the plots array for the current figure
            figPlots = figPar[plotsString]
            #Create a dictionary for the new figure (initially filled with default values)
            self.figures[figNum] = {numPlotsString: len(figPlots),
                                    plotsString:[{plotVarString: constants.DEFAULT_PLOT_VAR,
                                                  plotTypeString: defaultPlotType,
                                                  plotColorString: defaultPlotColor,
                                                  plotAlphaString: constants.DEFAULT_PLOT_ALPHA,
                                                  hasMinPlotValString: False,
                                                  hasMaxPlotValString: False,
                                                  minPlotValString: 0.0,
                                                  maxPlotValString: 0.0,
                                                  plotPlaneString: constants.XY_PLANE,
                                                  plotAxisString: constants.X_AXIS,
                                                  xPosString: 0.0,
                                                  yPosString: 0.0,
                                                  zPosString: 0.0,
                                                  }]*len(figPlots),
                                    axesString: [None]*len(figPlots),
                                    colorbarsString: [None]*len(figPlots),
                                    numRowsString: 1,
                                    numColsString: 1,
                                    figTitleString: "Figure " + str(figNum + 1),
                                    figFormatString: defaultSaveFigFormat,
                                    figDPIString: defaultSaveFigDPI,
                                    saveFigsString: defaultSaveFigs,
                                    makeMovieString: defaultMakeMovie}
            #Check if the user has provided a title for the figure
            if figTitleString in figPar.keys():
                self.figures[figNum][figTitleString] = figPar[figTitleString]
            #Check if the user wants to override the overall simPar saveFigs flag for this figure
            if saveFigsString in figPar.keys():
                self.figures[figNum][saveFigsString] = figPar[saveFigsString]
            #Check if the user wants to override the overall simPar makeMovie flag for this figure
            if makeMovieString in figPar.keys():
                self.figures[figNum][makeMovieString] = figPar[makeMovieString]
            #Check if the user wants this figure to have its own output file format
            if figFormatString in figPar.keys():
                self.figures[figNum][figFormatString] = figPar[figFormatString]
            #Check if the user wants this figure to have its own output image resolution
            if figDPIString in figPar.keys():
                self.figures[figNum][figDPIString] = figPar[figDPIString]
            #If we are saving images of this figure, we will need to check whether it has a valid file format.
            if self.figures[figNum][saveFigsString]:
                #If we are saving this figure, add its format to the file formats we need to check
                #(only if the file format isn't already in the list).
                if self.figures[figNum][figFormatString] not in imageFormatsToCheck:
                    imageFormatsToCheck.append(self.figures[figNum][figFormatString])
            #Check whether the user has specified the number of rows and columns that they want in the current
            #matplotlib figure. For example, 1 row with 2 columns is how you would display two data plots side-by-side.
            self.figures[figNum][numRowsString] = 1
            self.figures[figNum][numColsString] = 1
            #If the user has specified a number of rows and a number of columns, we can just use their values.
            if numRowsString in figPar.keys() and numColsString in figPar.keys():
                self.figures[figNum][numRowsString] = figPar[numRowsString]
                self.figures[figNum][numColsString] = figPar[numColsString]
            elif numRowsString in figPar.keys():
                #If the user has only provided the number of rows, we need to calculate a suitable number of columns
                self.figures[figNum][numRowsString] = figPar[numRowsString]
                #Calculate the number of columns that we need to accommodate all data plots
                #with the requested number of rows
                self.figures[figNum][numColsString] = np.ceil(self.figures[figNum][numPlotsString]
                                                              /self.figures[figNum][numRowsString])
            elif numColsString in figPar.keys():
                #If the user has only provided the number of columns, we need to calculate a suitable number of rows
                self.figures[figNum][numColsString] = figPar[numColsString]
                #Calculate the number of columns that we need to accommodate all data plots
                #with the requested number of rows
                self.figures[figNum][numRowsString] = np.ceil(self.figures[figNum][numPlotsString]
                                                              / self.figures[figNum][numColsString])
            else: #if the user has provided neither a number of rows nor a number of columns
                #Choose a number of rows that makes sense for the number of data plots
                if self.figures[figNum][numPlotsString] <= 2:
                    self.figures[figNum][numRowsString] = 1
                elif self.figures[figNum][numPlotsString] <= 4:
                    self.figures[figNum][numRowsString] = 2
                elif self.figures[figNum][numPlotsString] <= 9:
                    self.figures[figNum][numRowsString] = 3
                else: #shouldn't ever get here, unless the user overrides the default max number of plots per figure
                    self.figures[figNum][numRowsString] = 4
            #Calculate the number of columns that we need to accommodate all data plots with the derived number of rows
            self.figures[figNum][numColsString] = np.ceil(self.figures[figNum][numPlotsString]
                                                          /self.figures[figNum][numRowsString])
            #Iterate over the plots that the user wants to display in the figure
            for plotNum in range(self.figures[figNum][numPlotsString]):
                #Get the parameters for the current plot
                plotPar = figPlots[plotNum]
                #Create a dictionary for the current plot (initially filled with default values)
                self.figures[figNum][plotsString][plotNum] = {
                                                              plotVarString: constants.DEFAULT_PLOT_VAR,
                                                              plotTypeString: defaultPlotType,
                                                              plotColorString: defaultPlotColor,
                                                              plotAlphaString: constants.DEFAULT_PLOT_ALPHA,
                                                              hasMinPlotValString: False,
                                                              hasMaxPlotValString: False,
                                                              minPlotValString: 0.0,
                                                              maxPlotValString: 0.0,
                                                              plotPlaneString: constants.XY_PLANE,
                                                              plotAxisString: constants.X_AXIS,
                                                              xPosString: 0.0,
                                                              yPosString: 0.0,
                                                              zPosString: 0.0,
                                                              xPosPrecString: 1,
                                                              yPosPrecString: 1,
                                                              zPosPrecString: 1
                                                             }
                #Get the variable that the user wants to plot (e.g., density, pressure, Bx, etc.)
                if plotVarString in plotPar.keys():
                    self.figures[figNum][plotsString][plotNum][plotVarString] = plotPar[plotVarString]
                #Get the plot type (1D, 2D, or 3D)
                if plotTypeString in plotPar.keys():
                    self.figures[figNum][plotsString][plotNum][plotTypeString] = plotPar[plotTypeString]
                    #We might need to adjust the default plot color
                    if plotColorString not in plotPar.keys():
                        if self.figures[figNum][plotsString][plotNum][plotTypeString] == constants.PLOT_TYPE_1D:
                            self.figures[figNum][plotsString][plotNum][plotColorString] = constants.DEFAULT_PLOT_COLOR_1D
                        elif self.figures[figNum][plotsString][plotNum][plotTypeString] == constants.PLOT_TYPE_2D:
                            self.figures[figNum][plotsString][plotNum][plotColorString] = constants.DEFAULT_COLORMAP_2D
                        else:
                            self.figures[figNum][plotsString][plotNum][plotColorString] = constants.DEFAULT_COLORMAP_3D

                #Get the axis that they want to plot along (if they specified one)
                if plotAxisString in plotPar.keys():
                    self.figures[figNum][plotsString][plotNum][plotAxisString] = plotPar[plotAxisString]
                #Get the plane that they want to plot along (if they specified one)
                if plotPlaneString in plotPar.keys():
                    self.figures[figNum][plotsString][plotNum][plotPlaneString] = plotPar[plotPlaneString]
                #Get the minimum value of the variable that should be used for the plot's range
                #(by default it is slightly lower than the smallest value of the variable in the simulation grid)
                if minPlotValString in plotPar.keys():
                    self.figures[figNum][plotsString][plotNum][hasMinPlotValString] = True
                    self.figures[figNum][plotsString][plotNum][minPlotValString] = plotPar[minPlotValString]
                #Get the maximum value of the variable that should be used for the plot's range
                #(by default it is slightly higher than the largest value of the variable in the simulation grid)
                if maxPlotValString in plotPar.keys():
                    self.figures[figNum][plotsString][plotNum][hasMaxPlotValString] = True
                    self.figures[figNum][plotsString][plotNum][maxPlotValString] = plotPar[maxPlotValString]
                #Get the colour for the data plot
                if plotColorString in plotPar.keys():
                    self.figures[figNum][plotsString][plotNum][plotColorString] = plotPar[plotColorString]
                #Some parameters are only necessary for 2D and 3D simulations
                if simulationGrid.nDim == 2: #if the simulation is 2D
                    #If the plot type is a 1D line through the simulation grid, we need to know whether
                    #the line is parallel with the x- or y-axis and where is supposed to be located
                    #in the simulation volume.
                    if self.figures[figNum][plotsString][plotNum][plotTypeString] != constants.PLOT_TYPE_2D:
                        #If the line is parallel if the x-axis, we need to know its y-position.
                        if self.figures[figNum][plotsString][plotNum][plotAxisString] == constants.X_AXIS:
                            self.figures[figNum][plotsString][plotNum][yPosString] = plotPar[yPosString]
                            #Figure out how many decimal places we need to display the y-position in the plot title
                            yPos = plotPar[yPosString]
                            #Find the actual y-position that will get displayed
                            #(in case no cell is centred on exactly yPos)
                            yPosActual = yPos
                            if simulationGrid.yCoords.shape[0] == 1:
                                yPosActual = simulationGrid.yCoords[0,0]
                            else:
                                minY = simulationGrid.yCoords.min()
                                dy = simulationGrid.yCoords[1,0] - simulationGrid.yCoords[0,0]
                                index = int((yPos - minY)/dy)
                                yPosActual = simulationGrid.yCoords[index,0]
                            halfDyPrec = 1
                            while np.abs(np.power(10,halfDyPrec)*(0.5*simulationGrid.dy) - int(np.power(10,halfDyPrec)*0.5*simulationGrid.dy)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDyPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDyPrec += 1
                            minYPrec = 1
                            while np.abs(np.power(10,minYPrec)*(0.5*simulationGrid.minY) - int(np.power(10,minYPrec)*0.5*simulationGrid.minY)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minYPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minYPrec += 1
                            halfDyPrec = np.maximum(halfDyPrec, minYPrec)
                            yPosPrec = 1
                            while np.abs(np.power(10,yPosPrec)*yPosActual - int(np.power(10,yPosPrec)*yPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and yPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                yPosPrec += 1
                            yPosPrec = np.minimum(halfDyPrec, yPosPrec)
                            self.figures[figNum][plotsString][plotNum][yPosPrecString] = yPosPrec
                        else: #If the line is parallel if the y-axis, we need to know its x-position.
                            self.figures[figNum][plotsString][plotNum][xPosString] = plotPar[xPosString]
                            #Figure out how many decimal places we need to display the x-position in the plot title
                            xPos = plotPar[xPosString]
                            #Find the actual x-position that will get displayed
                            #(in case no cell is centred on exactly xPos)
                            xPosActual = xPos
                            if simulationGrid.xCoords.shape[1] == 1:
                                xPosActual = simulationGrid.xCoords[0,0]
                            else:
                                minX = simulationGrid.xCoords.min()
                                dx = simulationGrid.xCoords[0,1] - simulationGrid.xCoords[0,0]
                                index = int((xPos - minX)/dx)
                                xPosActual = simulationGrid.xCoords[0,index]
                            halfDxPrec = 1
                            while np.abs(np.power(10,halfDxPrec)*(0.5*simulationGrid.dx) - int(np.power(10,halfDxPrec)*0.5*simulationGrid.dx)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDxPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDxPrec += 1
                            minXPrec = 1
                            while np.abs(np.power(10,minXPrec)*(0.5*simulationGrid.minX) - int(np.power(10,minXPrec)*0.5*simulationGrid.minX)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minXPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minXPrec += 1
                            halfDxPrec = np.maximum(halfDxPrec, minXPrec)
                            xPosPrec = 1
                            while np.abs(np.power(10,xPosPrec)*xPosActual - int(np.power(10,xPosPrec)*xPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and xPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                xPosPrec += 1
                            xPosPrec = np.minimum(halfDxPrec, xPosPrec)
                            self.figures[figNum][plotsString][plotNum][xPosPrecString] = xPosPrec
                elif simulationGrid.nDim == 3: #if the simulation is 3D
                    #Get the plot type (could be 1D, 2D, or 3D for a 3D sim)
                    if plotTypeString in plotPar.keys():
                        self.figures[figNum][plotsString][plotNum][plotTypeString] = plotPar[plotTypeString]
                    #If the plot type is a 1D line through the simulation grid, we need to know whether
                    #the line is parallel with the x-, y-, or z-axis and where is supposed to be located
                    #in the simulation volume.
                    if self.figures[figNum][plotsString][plotNum][plotTypeString] == constants.PLOT_TYPE_1D:
                        self.figures[figNum][plotsString][plotNum][plotAxisString] = plotPar[plotAxisString]
                        #If the line is parallel with the x-axis, we need to know its y- and z-position.
                        if self.figures[figNum][plotsString][plotNum][plotAxisString] == constants.X_AXIS:
                            self.figures[figNum][plotsString][plotNum][yPosString] = plotPar[yPosString]
                            #Figure out how many decimal places we need to display the y-position in the plot title
                            yPos = plotPar[yPosString]
                            #Find the actual y-position that will get displayed
                            #(in case no cell is centred on exactly yPos)
                            yPosActual = yPos
                            if simulationGrid.yCoords.shape[0] == 1:
                                yPosActual = simulationGrid.yCoords[0,0,0]
                            else:
                                minY = simulationGrid.yCoords.min()
                                dy = simulationGrid.yCoords[1,0,0] - simulationGrid.yCoords[0,0,0]
                                index = int((yPos - minY)/dy)
                                yPosActual = simulationGrid.yCoords[index,0,0]
                            halfDyPrec = 1
                            while np.abs(np.power(10,halfDyPrec)*(0.5*simulationGrid.dy) - int(np.power(10,halfDyPrec)*0.5*simulationGrid.dy)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDyPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDyPrec += 1
                            minYPrec = 1
                            while np.abs(np.power(10,minYPrec)*(0.5*simulationGrid.minY) - int(np.power(10,minYPrec)*0.5*simulationGrid.minY)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minYPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minYPrec += 1
                            halfDyPrec = np.maximum(halfDyPrec, minYPrec)
                            yPosPrec = 1
                            while np.abs(np.power(10,yPosPrec)*yPosActual - int(np.power(10,yPosPrec)*yPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and yPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                yPosPrec += 1
                            yPosPrec = np.minimum(halfDyPrec, yPosPrec)
                            self.figures[figNum][plotsString][plotNum][yPosPrecString] = yPosPrec
                            self.figures[figNum][plotsString][plotNum][zPosString] = plotPar[zPosString]
                            #Figure out how many decimal places we need to display the z-position in the plot title
                            zPos = plotPar[zPosString]
                            #Find the actual z-position that will get displayed
                            #(in case no cell is centred on exactly zPos)
                            zPosActual = zPos
                            if simulationGrid.xCoords.shape[2] == 1:
                                zPosActual = simulationGrid.zCoords[0,0,0]
                            else:
                                minZ = simulationGrid.zCoords.min()
                                dz = simulationGrid.zCoords[0,0,1] - simulationGrid.zCoords[0,0,0]
                                index = int((zPos - minZ)/dz)
                                zPosActual = simulationGrid.zCoords[0,0,index]
                            halfDzPrec = 1
                            while np.abs(np.power(10,halfDzPrec)*(0.5*simulationGrid.dz) - int(np.power(10,halfDzPrec)*0.5*simulationGrid.dz)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDzPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDzPrec += 1
                            minZPrec = 1
                            while np.abs(np.power(10,minZPrec)*(0.5*simulationGrid.minZ) - int(np.power(10,minZPrec)*0.5*simulationGrid.minZ)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minZPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minZPrec += 1
                            halfDzPrec = np.maximum(halfDzPrec, minZPrec)
                            zPosPrec = 1
                            while np.abs(np.power(10,zPosPrec)*zPosActual - int(np.power(10,zPosPrec)*zPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and zPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                zPosPrec += 1
                            zPosPrec = np.minimum(halfDzPrec, zPosPrec)
                            self.figures[figNum][plotsString][plotNum][zPosPrecString] = zPosPrec
                        elif self.figures[figNum][plotsString][plotNum][plotAxisString] == constants.Y_AXIS:
                            #If the line is parallel with the y-axis, we need to know its x- and z-position.
                            self.figures[figNum][plotsString][plotNum][xPosString] = plotPar[xPosString]
                            #Figure out how many decimal places we need to display the x-position in the plot title
                            xPos = plotPar[xPosString]
                            #Find the actual x-position that will get displayed
                            #(in case no cell is centred on exactly xPos)
                            xPosActual = xPos
                            if simulationGrid.xCoords.shape[1] == 1:
                                xPosActual = simulationGrid.xCoords[0,0,0]
                            else:
                                minX = simulationGrid.xCoords.min()
                                dx = simulationGrid.xCoords[0,1,0] - simulationGrid.xCoords[0,0,0]
                                index = int((xPos - minX)/dx)
                                xPosActual = simulationGrid.xCoords[0,index,0]
                            halfDxPrec = 1
                            while np.abs(np.power(10,halfDxPrec)*(0.5*simulationGrid.dx) - int(np.power(10,halfDxPrec)*0.5*simulationGrid.dx)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDxPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDxPrec += 1
                            minXPrec = 1
                            while np.abs(np.power(10,minXPrec)*(0.5*simulationGrid.minX) - int(np.power(10,minXPrec)*0.5*simulationGrid.minX)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minXPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minXPrec += 1
                            halfDxPrec = np.maximum(halfDxPrec, minXPrec)
                            xPosPrec = 1
                            while np.abs(np.power(10,xPosPrec)*xPosActual - int(np.power(10,xPosPrec)*xPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and xPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                xPosPrec += 1
                            xPosPrec = np.minimum(halfDxPrec, xPosPrec)
                            self.figures[figNum][plotsString][plotNum][xPosPrecString] = xPosPrec
                            self.figures[figNum][plotsString][plotNum][zPosString] = plotPar[zPosString]
                            #Figure out how many decimal places we need to display the z-position in the plot title
                            zPos = plotPar[zPosString]
                            #Find the actual z-position that will get displayed
                            #(in case no cell is centred on exactly zPos)
                            zPosActual = zPos
                            if simulationGrid.xCoords.shape[2] == 1:
                                zPosActual = simulationGrid.zCoords[0,0,0]
                            else:
                                minZ = simulationGrid.zCoords.min()
                                dz = simulationGrid.zCoords[0,0,1] - simulationGrid.zCoords[0,0,0]
                                index = int((zPos - minZ)/dz)
                                zPosActual = simulationGrid.zCoords[0,0,index]
                            halfDzPrec = 1
                            while np.abs(np.power(10,halfDzPrec)*(0.5*simulationGrid.dz) - int(np.power(10,halfDzPrec)*0.5*simulationGrid.dz)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDzPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDzPrec += 1
                            minZPrec = 1
                            while np.abs(np.power(10,minZPrec)*(0.5*simulationGrid.minZ) - int(np.power(10,minZPrec)*0.5*simulationGrid.minZ)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minZPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minZPrec += 1
                            halfDzPrec = np.maximum(halfDzPrec, minZPrec)
                            zPosPrec = 1
                            while np.abs(np.power(10,zPosPrec)*zPosActual - int(np.power(10,zPosPrec)*zPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and zPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                zPosPrec += 1
                            zPosPrec = np.minimum(halfDzPrec, zPosPrec)
                            self.figures[figNum][plotsString][plotNum][zPosPrecString] = zPosPrec
                        else: #If the line is parallel with the z-axis, we need to know its x- and y-position.
                            self.figures[figNum][plotsString][plotNum][xPosString] = plotPar[xPosString]
                            #Figure out how many decimal places we need to display the x-position in the plot title
                            xPos = plotPar[xPosString]
                            #Find the actual x-position that will get displayed
                            #(in case no cell is centred on exactly xPos)
                            xPosActual = xPos
                            if simulationGrid.xCoords.shape[1] == 1:
                                xPosActual = simulationGrid.xCoords[0,0,0]
                            else:
                                minX = simulationGrid.xCoords.min()
                                dx = simulationGrid.xCoords[0,1,0] - simulationGrid.xCoords[0,0,0]
                                index = int((xPos - minX)/dx)
                                xPosActual = simulationGrid.xCoords[0,index,0]
                            halfDxPrec = 1
                            while np.abs(np.power(10,halfDxPrec)*(0.5*simulationGrid.dx) - int(np.power(10,halfDxPrec)*0.5*simulationGrid.dx)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDxPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDxPrec += 1
                            minXPrec = 1
                            while np.abs(np.power(10,minXPrec)*(0.5*simulationGrid.minX) - int(np.power(10,minXPrec)*0.5*simulationGrid.minX)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minXPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minXPrec += 1
                            halfDxPrec = np.maximum(halfDxPrec, minXPrec)
                            xPosPrec = 1
                            while np.abs(np.power(10,xPosPrec)*xPosActual - int(np.power(10,xPosPrec)*xPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and xPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                xPosPrec += 1
                            xPosPrec = np.minimum(halfDxPrec, xPosPrec)
                            self.figures[figNum][plotsString][plotNum][xPosPrecString] = xPosPrec
                            self.figures[figNum][plotsString][plotNum][yPosString] = plotPar[yPosString]
                            #Figure out how many decimal places we need to display the y-position in the plot title
                            yPos = plotPar[yPosString]
                            #Find the actual y-position that will get displayed
                            #(in case no cell is centred on exactly yPos)
                            yPosActual = yPos
                            if simulationGrid.yCoords.shape[0] == 1:
                                yPosActual = simulationGrid.yCoords[0,0,0]
                            else:
                                minY = simulationGrid.yCoords.min()
                                dy = simulationGrid.yCoords[1,0,0] - simulationGrid.yCoords[0,0,0]
                                index = int((yPos - minY)/dy)
                                yPosActual = simulationGrid.yCoords[index,0,0]
                            halfDyPrec = 1
                            while np.abs(np.power(10,halfDyPrec)*(0.5*simulationGrid.dy) - int(np.power(10,halfDyPrec)*0.5*simulationGrid.dy)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDyPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDyPrec += 1
                            minYPrec = 1
                            while np.abs(np.power(10,minYPrec)*(0.5*simulationGrid.minY) - int(np.power(10,minYPrec)*0.5*simulationGrid.minY)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minYPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minYPrec += 1
                            halfDyPrec = np.maximum(halfDyPrec, minYPrec)
                            yPosPrec = 1
                            while np.abs(np.power(10,yPosPrec)*yPosActual - int(np.power(10,yPosPrec)*yPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and yPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                yPosPrec += 1
                            yPosPrec = np.minimum(halfDyPrec, yPosPrec)
                            self.figures[figNum][plotsString][plotNum][yPosPrecString] = yPosPrec
                    elif self.figures[figNum][plotsString][plotNum][plotTypeString] == constants.PLOT_TYPE_2D:
                        #If the plot is a 2D cross-section, we need to know whether the cross-section should be
                        #parallel with the xy-,yz-, or xz-plane and where it is supposed to be located in the
                        #simulation volume.
                        self.figures[figNum][plotsString][plotNum][plotPlaneString] = plotPar[plotPlaneString]
                        #If the cross-section is parallel with the xy-plane, we need to know its z-position.
                        if self.figures[figNum][plotsString][plotNum][plotPlaneString] == constants.XY_PLANE:
                            self.figures[figNum][plotsString][plotNum][zPosString] = plotPar[zPosString]
                            #Figure out how many decimal places we need to display the z-position in the plot title
                            zPos = plotPar[zPosString]
                            #Find the actual z-position that will get displayed
                            #(in case no cell is centred on exactly zPos)
                            zPosActual = zPos
                            if simulationGrid.xCoords.shape[2] == 1:
                                zPosActual = simulationGrid.zCoords[0,0,0]
                            else:
                                minZ = simulationGrid.zCoords.min()
                                dz = simulationGrid.zCoords[0,0,1] - simulationGrid.zCoords[0,0,0]
                                index = int((zPos - minZ)/dz)
                                zPosActual = simulationGrid.zCoords[0,0,index]
                            halfDzPrec = 1
                            while np.abs(np.power(10,halfDzPrec)*(0.5*simulationGrid.dz) - int(np.power(10,halfDzPrec)*0.5*simulationGrid.dz)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDzPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDzPrec += 1
                            minZPrec = 1
                            while np.abs(np.power(10,minZPrec)*(0.5*simulationGrid.minZ) - int(np.power(10,minZPrec)*0.5*simulationGrid.minZ)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minZPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minZPrec += 1
                            halfDzPrec = np.maximum(halfDzPrec, minZPrec)
                            zPosPrec = 1
                            while np.abs(np.power(10,zPosPrec)*zPosActual - int(np.power(10,zPosPrec)*zPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and zPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                zPosPrec += 1
                            zPosPrec = np.minimum(halfDzPrec, zPosPrec)
                            self.figures[figNum][plotsString][plotNum][zPosPrecString] = zPosPrec
                        elif self.figures[figNum][plotsString][plotNum][plotPlaneString] == constants.XZ_PLANE:
                            #If the cross-section is parallel with the xz-plane, we need to know its y-position.
                            self.figures[figNum][plotsString][plotNum][yPosString] = plotPar[yPosString]
                            #Figure out how many decimal places we need to display the y-position in the plot title
                            yPos = plotPar[yPosString]
                            #Find the actual y-position that will get displayed
                            #(in case no cell is centred on exactly yPos)
                            yPosActual = yPos
                            if simulationGrid.yCoords.shape[0] == 1:
                                yPosActual = simulationGrid.yCoords[0,0,0]
                            else:
                                minY = simulationGrid.yCoords.min()
                                dy = simulationGrid.yCoords[1,0,0] - simulationGrid.yCoords[0,0,0]
                                index = int((yPos - minY)/dy)
                                yPosActual = simulationGrid.yCoords[index,0]
                            halfDyPrec = 1
                            while np.abs(np.power(10,halfDyPrec)*(0.5*simulationGrid.dy) - int(np.power(10,halfDyPrec)*0.5*simulationGrid.dy)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDyPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDyPrec += 1
                            minYPrec = 1
                            while np.abs(np.power(10,minYPrec)*(0.5*simulationGrid.minY) - int(np.power(10,minYPrec)*0.5*simulationGrid.minY)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minYPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minYPrec += 1
                            halfDyPrec = np.maximum(halfDyPrec, minYPrec)
                            yPosPrec = 1
                            while np.abs(np.power(10,yPosPrec)*yPosActual - int(np.power(10,yPosPrec)*yPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and yPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                yPosPrec += 1
                            yPosPrec = np.minimum(halfDyPrec, yPosPrec)
                            self.figures[figNum][plotsString][plotNum][yPosPrecString] = yPosPrec
                        else: #If the cross-section is parallel with the yz-plane, we need to know its x-position.
                            self.figures[figNum][plotsString][plotNum][xPosString] = plotPar[xPosString]
                            #Figure out how many decimal places we need to display the x-position in the plot title
                            xPos = plotPar[xPosString]
                            #Find the actual x-position that will get displayed
                            #(in case no cell is centred on exactly xPos)
                            xPosActual = xPos
                            if simulationGrid.xCoords.shape[1] == 1:
                                xPosActual = simulationGrid.xCoords[0,0,0]
                            else:
                                minX = simulationGrid.xCoords.min()
                                dx = simulationGrid.xCoords[0,1,0] - simulationGrid.xCoords[0,0,0]
                                index = int((xPos - minX)/dx)
                                xPosActual = simulationGrid.xCoords[0,index,0]
                            halfDxPrec = 1
                            while np.abs(np.power(10,halfDxPrec)*(0.5*simulationGrid.dx) - int(np.power(10,halfDxPrec)*0.5*simulationGrid.dx)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and halfDxPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                halfDxPrec += 1
                            minXPrec = 1
                            while np.abs(np.power(10,minXPrec)*(0.5*simulationGrid.minX) - int(np.power(10,minXPrec)*0.5*simulationGrid.minX)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and minXPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                minXPrec += 1
                            halfDxPrec = np.maximum(halfDxPrec, minXPrec)
                            xPosPrec = 1
                            while np.abs(np.power(10,xPosPrec)*xPosActual - int(np.power(10,xPosPrec)*xPosActual)) > constants.TIME_PRECISION_SMALL_VAL \
                                    and xPosPrec < constants.MAX_PLOT_TITLE_POSITION_PREC:
                                xPosPrec += 1
                            xPosPrec = np.minimum(halfDxPrec, xPosPrec)
                            self.figures[figNum][plotsString][plotNum][xPosPrecString] = xPosPrec
                    else:
                        #If the plot is a 3D colormap, we need to check whether the user wants to use a particular
                        #alpha value for the plot's transparency. (The default value in PythonMHD_Constants.py
                        #will be used if the user does not provide a plotAlpha value in visPar.)
                        if plotAlphaString in plotPar.keys():
                            self.figures[figNum][plotsString][plotNum][plotAlphaString] = plotPar[plotAlphaString]
        #If we are saving images of the matplotlib figures, we need to make sure that the requested output format
        #(e.g., "png", "pdf", etc.) is available in the matplotlib backend.
        if len(imageFormatsToCheck) > 0:
            #Create a test figure for getting the supported file types
            testFigure = plt.figure()
            #Get the available file types
            availableFormats = testFigure.canvas.get_supported_filetypes()
            #Close the test figure
            plt.close()
            #Make sure that every format the user has requested is in our list of available file types
            for format in imageFormatsToCheck:
                if format not in availableFormats.keys():
                    raise RuntimeError("\nVISUALIZATION ERROR: You have requested a file format (" + format + ") "
                                       + "in visPar that is not supported by your matplotlib backend ("
                                       + matplotlibBackend + ").\nThe image formats that this backend supports are "
                                       + "the following:\n" + str(availableFormats))
        for figNum in range(self.numFigures):
            #Adjust the size of the figure based on the number of rows and columns
            if self.figures[figNum][numColsString] == 1:
                self.figures[figNum][figString] = plt.figure(figsize=(6.5*self.figures[figNum][numColsString],
                                                                      4.5 + 4.5*(self.figures[figNum][numRowsString] - 1)))
            else:
                self.figures[figNum][figString] = plt.figure(figsize=(6.5*self.figures[figNum][numColsString],
                                                                      4.5 + 3.5*(self.figures[figNum][numRowsString]-1)))
            #Set the title for the figure
            self.figures[figNum][figString].suptitle(self.figures[figNum][figTitleString],fontsize=16)
            #Now we will create all of the plots that we need to display on the figure
            for plotNum in range(self.figures[figNum][numPlotsString]):
                #Get the parameters for the current plot
                plotPar = self.figures[figNum][plotsString][plotNum]
                #Get the plot's type
                plotType = plotPar[plotTypeString]
                #Get the variable it is supposed to display
                plotVar = plotPar[plotVarString]
                #Create a suitable matplotlib subplot based on the plot type
                if plotType == constants.PLOT_TYPE_3D:
                    #For 3D visualizations, we need to explicitly tell matplotlib that the subplot is 3D.
                    self.figures[figNum][axesString][plotNum] = self.figures[figNum][figString].add_subplot(
                                                        self.figures[figNum][numRowsString],
                                                        self.figures[figNum][numColsString],plotNum+1,projection='3d')
                else:
                    #We can use the same subplot creation call for 1D and 2D plots.
                    self.figures[figNum][axesString][plotNum] = self.figures[figNum][figString].add_subplot(
                                                        self.figures[figNum][numRowsString],
                                                        self.figures[figNum][numColsString],plotNum+1)
                #Get the axis object for the current plot
                plotAxis = self.figures[figNum][axesString][plotNum]
                #Check whether the axis is a 1D, 2D or 3D plot
                if plotType == constants.PLOT_TYPE_1D:
                    #If the plot is 1D, get the axis (x, y, or z) that is parallel with the line of data.
                    plotAxisType = plotPar[plotAxisString]
                    #Set the label for the y-axis of the plot
                    plotAxis.set_ylabel(plotVar)
                    #Determine the correct label and range for the x-axis
                    if plotAxisType == constants.X_AXIS:
                        plotAxis.set_xlim(simulationGrid.minX, simulationGrid.maxX)
                        plotAxis.set_xlabel("x-position")
                    elif plotAxisType == constants.Y_AXIS:
                        plotAxis.set_xlim(simulationGrid.minY, simulationGrid.maxY)
                        plotAxis.set_xlabel("y-position")
                    else:
                        plotAxis.set_xlim(simulationGrid.minZ, simulationGrid.maxZ)
                        plotAxis.set_xlabel("z-position")
                elif plotType == constants.PLOT_TYPE_2D:
                    #If the plot is 2D, we need to know which plane (xy, yz, or xz) is parallel with the data.
                    plotPlane = plotPar[plotPlaneString]
                    #Using the plane, determine the correct labels and ranges for the x- and y-axes.
                    if plotPlane == constants.XY_PLANE:
                        plotAxis.set_xlim(simulationGrid.minX, simulationGrid.maxX)
                        plotAxis.set_xlabel("x-position")
                        plotAxis.set_ylim(simulationGrid.minY, simulationGrid.maxY)
                        plotAxis.set_ylabel("y-position")
                    elif plotAxis == constants.YZ_PLANE:
                        plotAxis.set_xlim(simulationGrid.minY, simulationGrid.maxY)
                        plotAxis.set_xlabel("y-position")
                        plotAxis.set_ylim(simulationGrid.minZ, simulationGrid.maxZ)
                        plotAxis.set_ylabel("z-position")
                    else:
                        plotAxis.set_xlim(simulationGrid.minX, simulationGrid.maxX)
                        plotAxis.set_xlabel("x-position")
                        plotAxis.set_ylim(simulationGrid.minZ, simulationGrid.maxZ)
                        plotAxis.set_ylabel("z-position")
                else: #if the plot is 3D
                    #Set the labels and ranges for the x, y, and z-axes
                    plotAxis.set_xlim(simulationGrid.minX, simulationGrid.maxX)
                    plotAxis.set_xlabel("x-position")
                    plotAxis.set_ylim(simulationGrid.minY, simulationGrid.maxY)
                    plotAxis.set_ylabel("y-position")
                    plotAxis.set_zlim(simulationGrid.minZ, simulationGrid.maxZ)
                    plotAxis.set_zlabel("z-position")
            #After creating all of the plots, tighten the layout to maximize the size of each plot.
            self.figures[figNum][figString].tight_layout()
            #Set the current figure (so matplotlib knows which figure we are currently adjusting)
            plt.figure(self.figures[figNum][figString].number)
            #Apply some further adjustments (for visual appeal) based on the number of rows and columns
            plt.subplots_adjust(left=0.075, wspace=0.6, right=0.9)
            if self.figures[figNum][numRowsString] == 1:
                plt.subplots_adjust(top=0.88)
            else:
                plt.subplots_adjust(top=0.93, hspace=0.44)
            if self.figures[figNum][numColsString] == 1:
                plt.subplots_adjust(left=0.125, right=0.8)
                # if self.figures[figNum][constants.PLOTS][0][constants.PLOT_TYPE] == constants.PLOT_TYPE_3D:
                #     plt.subplots_adjust(right=0.8)
                # if self.figures[figNum][numRowsString] > 1:
                #     plt.subplots_adjust(left=-0.5)
                # else:
                #     test = 0
                #     #plt.subplots_adjust(left=0.5)
            # plt.subplots_adjust(hspace=0.3,wspace=0.4,top=0.94,left=0.06,bottom=0.07,right=0.95)

    ######PRIVATE METHODS########
    #Function: __getPlotData
    #Purpose: Retrieves the data from the simulationGrid that the user wants to plot.
    #Inputs: plotVarString (the string for the variable that should be retrieved from the grid)
    #        simulationGrid (the SimulationGrid object that contains all of the simulation data)
    #Outputs: plotData (the data that should be used for the plot)
    def __getPlotData(self, plotVarString, simulationGrid):
        #Use the plotVarString to determine which variable we should retrieve from the simulation grid
        if plotVarString == constants.DENSITY: #if the variable is density
            plotData = simulationGrid.primVars[0]
        elif plotVarString == constants.VX: #if the variable is x-velocity
            plotData = simulationGrid.primVars[1]
        elif plotVarString == constants.VY: #if the variable is y-velocity
            plotData = simulationGrid.primVars[2]
        elif plotVarString == constants.VZ: #if the variable is z-velocity
            plotData = simulationGrid.primVars[3]
        elif plotVarString == constants.V: #if the variable is total velocity
            plotData = np.sqrt(np.square(simulationGrid.primVars[1]) + np.square(simulationGrid.primVars[2])
                               + np.square(simulationGrid.primVars[3]))
        elif plotVarString == constants.V2: #if the variable is total velocity squared
            plotData = np.square(simulationGrid.primVars[1]) + np.square(simulationGrid.primVars[2]) \
                       + np.square(simulationGrid.primVars[3])
        elif plotVarString == constants.PRESSURE: #if the variable is hydrodynamic pressure
            if simulationGrid.isMHD:
                plotData = simulationGrid.primVars[7]
            else:
                plotData = simulationGrid.primVars[4]
        elif plotVarString == constants.ENERGY: #if the variable is energy
            if simulationGrid.isMHD:
                plotData = simulationGrid.consVars[7]
            else:
                plotData = simulationGrid.consVars[4]
        elif plotVarString == constants.BX: #if the variable is Bx (x-component of the magnetic field
            plotData = simulationGrid.primVars[4]
        elif plotVarString == constants.BY: #if the variable is By (y-component of the magnetic field)
            plotData = simulationGrid.primVars[5]
        elif plotVarString == constants.BZ: #if the variable is Bz (z-component of the magnetic field)
            plotData = simulationGrid.primVars[6]
        elif plotVarString == constants.B: #if the variable is B (total magnetic field intensity)
            plotData = np.sqrt(np.square(simulationGrid.primVars[4]) + np.square(simulationGrid.primVars[5])
                               + np.square(simulationGrid.primVars[6]))
        elif plotVarString == constants.B2: #if the variable is B^2 (total magnetic field intensity squared)
            plotData = np.square(simulationGrid.primVars[4]) + np.square(simulationGrid.primVars[5]) \
                       + np.square(simulationGrid.primVars[6])
        else: #if the variable is total pressure (hydrodynamic pressure + magnetic pressure)
            plotData = simulationGrid.primVars[7] + (np.square(simulationGrid.primVars[4])
                                                     + np.square(simulationGrid.primVars[5])
                                                     + np.square(simulationGrid.primVars[6]))/2.0
        #Return the simulation data for the plot
        return plotData

    #Function: __update1DPlot
    #Purpose: Updates a 1D line plot of simulation data.
    #Inputs: axis (the axis object for the plot)
    #        coords (an array with the positions/coordinates for all of the data points)
    #        plotVar (the string for the variable that the plot visualizes)
    #        plotData (the data that should be shown on the updated plot)
    #        plotColor (the color that should be used for the plot)
    #        axisType (the axis (x, y, or z) for the coords/positions)
    #Outputs: None (but the changes will show up in the matplotlib figure)
    def __update1DPlot(self, axis, coords, plotVar, plotData, plotColor, axisType):
        #Clear the old data
        axis.clear()
        #Add the new data
        axis.plot(coords, plotData, color=plotColor)
        #Set the position range
        axis.set_xlim(coords.min(), coords.max())
        #Set the correct label (x-, y-, or z-position) for the x-axis
        if axisType == constants.X_AXIS:
            axis.set_xlabel("x-position")
        elif axisType == constants.Y_AXIS:
            axis.set_xlabel("y-position")
        else:
            axis.set_xlabel("z-position")
        #Set the y-axis label
        axis.set_ylabel(plotVar)

    #Function: __update2DPlot
    #Purpose: Updates a 2D colormap plot of simulation data.
    #Inputs: figNum (the number for the figure that the plot belongs to)
    #        plotNum (the number for the plot)
    #        axis (the axis object for the plot)
    #        simulationGrid (the SimulationGrid object that contains the simulation data)
    #        plotData (the data that should be shown on the updated plot)
    #        plotColor (the color that should be used for the plot)
    #        horizAxisType (the axis (x, y, or z) that should be displayed on the horizontal axis of the plot)
    #        vertAxisType (the axis (x, y, or z) that should be displayed on the vertical axis of the plot)
    #        minPlotVal (the minimum value of the simulation data that should be used for the colormap)
    #        maxPlotVal (the maximum value of the simulation data that should be used for the colormap)
    #Outputs: None (but the changes will show up in the matplotlib figure)
    def __update2DPlot(self, figNum, plotNum, axis, simulationGrid, plotData, plotColor, horizAxisType, vertAxisType,
                           minPlotVal, maxPlotVal):
        #Clear the old simulation data
        axis.clear()
        #Create a 2D colormap of the new data
        if maxPlotVal - minPlotVal < constants.MINIMUM_COLORMAP_DIFFERENCE:
            img = axis.imshow(plotData,extent=[simulationGrid.minX, simulationGrid.maxX,
                                               simulationGrid.minY, simulationGrid.maxY],
                                       cmap=plotColor,vmin=minPlotVal-0.1,vmax=maxPlotVal+0.1)
        else:
            img = axis.imshow(plotData,extent=[simulationGrid.minX, simulationGrid.maxX,
                                               simulationGrid.minY, simulationGrid.maxY],
                                 cmap=plotColor,vmin=minPlotVal,vmax=maxPlotVal)
        #Set the correct labels for the horizontal and vertical axes
        if horizAxisType == constants.X_AXIS:
            axis.set_xlabel("x-position")
        elif horizAxisType == constants.Y_AXIS:
            axis.set_xlabel("y-position")
        else:
            axis.set_xlabel("z-position")
        if vertAxisType == constants.X_AXIS:
            axis.set_ylabel("x-position")
        elif vertAxisType == constants.Y_AXIS:
            axis.set_ylabel("y-position")
        else:
            axis.set_ylabel("z-position")
        #Make the axis object locatable so we can more easily position the colorbar for the colormap
        locatableAxis = make_axes_locatable(axis)
        #Append the colorbar axis to the plot axis
        colorbarAxis = locatableAxis.append_axes('right','5%','5%')
        #Put the colorbar in the colorbarAxis object
        colorbar = self.figures[figNum][constants.FIGURE].colorbar(img,cax=colorbarAxis)
        #Add the colorbar object to the SimVisualizer
        self.figures[figNum][constants.COLORBARS][plotNum] = colorbar

    #Function: __update3DPlot
    #Purpose: Updates a 3D colormap plot of simulation data.
    #Inputs: figNum (the number for the figure that the plot belongs to)
    #        plotNum (the number for the plot)
    #        axis (the axis object for the plot)
    #        simulationGrid (the SimulationGrid object that contains the simulation data)
    #        plotData (the data that should be shown on the updated plot)
    #        plotColor (the color that should be used for the plot)
    #        plotAlpha (the alpha/transparency value that should be used for the plot)
    #        hasMinPlotVal (a boolean flag for whether there is a user-defined minimum value for the colormap)
    #        userMinPlotVal (the user's minimum value for the colormap)
    #        hasMaxPlotVal (a boolean flag for whether there is a user-defined maximum value for the plot)
    #        userMaxPlotVal (the user's maximum value for the colormap)
    #Outputs: None (but the changes will show up in the matplotlib figure)
    def __update3DPlot(self, figNum, plotNum, axis, simulationGrid, plotData, plotColor, plotAlpha,hasMinPlotVal,
                           userMinPlotVal,hasMaxPlotVal,userMaxPlotVal):
        axis.clear()
        minPlotVal = plotData.min()
        maxPlotVal = plotData.max()
        if hasMinPlotVal:
            minPlotVal = userMinPlotVal
        if hasMaxPlotVal:
            maxPlotVal = userMaxPlotVal
        img = axis.scatter3D(simulationGrid.xCoords,
                       simulationGrid.yCoords,
                       simulationGrid.zCoords,
                       alpha=plotAlpha,
                       c=plotData.reshape(-1),
                       cmap=plotColor,vmin=minPlotVal, vmax=maxPlotVal)#vmin=minPlotVal, vmax=maxPlotVal
        axis.set_xlabel("x-position")
        axis.set_ylabel("y-position")
        axis.set_zlabel("z-position")
        plt.sca(axis)
        locatableAxis = make_axes_locatable(axis)
        figure = self.figures[figNum][constants.FIGURE]
        colorbarAxis = figure.add_axes([axis.get_position().x1 + 0.03,
                                          axis.get_position().y0,
                                          0.02,
                                          axis.get_position().height])
        #colorbar = self.fig.colorbar(img,cax=colorbarAxis)
        colorbar = figure.colorbar(img,cax=colorbarAxis)
        colorbar.set_alpha(1)
        colorbar.draw_all()
        self.figures[figNum][constants.COLORBARS][plotNum] = colorbar

    #Function: __getXAxisSlice_2D
    #Purpose: Retrieves the data along a particular line (parallel with the x-axis) in a 2D simulation.
    #Inputs: plotData (the 2D simulation data)
    #        yCoords (the y-coordinates of the cells in the simulation grid)
    #        yPos (the y-position of the horizontal/x-axis slice)
    #Outputs: lineData (the plot data along the requested horizontal line)
    #         yPos_actual (the actual y-position of the horizontal slice/line)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      y-position that was requested by the user)
    def __getXAxisSlice_2D(self, plotData, yCoords, yPos):
        #Ensure that the simulation data is 2D
        assert(len(plotData.shape) == 2)
        #Determine the y-position that is the closest to the y-position that the user has requested
        if yCoords.shape[0] == 1:
            lineData = plotData.reshape(-1)
            yPos_actual = yCoords[0,0]
        else:
            minY = yCoords.min()
            dy = yCoords[1,0] - yCoords[0,0]
            index = int((yPos - minY)/dy)
            assert(0 <= index < plotData.shape[0])
            lineData = plotData[index,:].reshape(-1)
            yPos_actual = yCoords[index,0]
        #Return the line of data and its actual y-position
        return lineData, yPos_actual

    #Function: __getYAxisSlice_2D
    #Purpose: Retrieves the data along a particular line (parallel with the y-axis) in a 2D simulation.
    #Inputs: plotData (the 2D simulation data)
    #        xCoords (the x-coordinates of the cells in the simulation grid)
    #        xPos (the x-position of the vertical/y-axis slice)
    #Outputs: lineData (the plot data along the requested vertical line)
    #         xPos_actual (the actual x-position of the vertical slice/line)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      x-position that was requested by the user)
    def __getYAxisSlice_2D(self, plotData, xCoords, xPos):
        #Ensure that the simulation data is 2D
        assert(len(plotData.shape) == 2)
        #Determine the x-position that is the closest to the x-position that the user has requested
        if xCoords.shape[1] == 1:
            newPlotData = plotData.reshape(-1)
            xPos_actual = xCoords[0,0]
        else:
            minX = xCoords.min()
            dx = xCoords[0,1] - xCoords[0,0]
            index = int((xPos - minX)/dx)
            assert(0 <= index < plotData.shape[1])
            newPlotData = plotData[:,index].reshape(-1)
            xPos_actual = xCoords[0,index]
        #Return the line of data and its actual x-position
        return newPlotData, xPos_actual

    #Function: __getXAxisSlice_3D
    #Purpose: Retrieves the data along a particular line (parallel with the x-axis) in a 3D simulation.
    #Inputs: plotData (the 3D simulation data)
    #        yCoords (the y-coordinates of the cells in the simulation grid)
    #        zCoords (the z-coordinates of the cells in the simulation grid)
    #        yPos (the y-position of the horizontal/x-axis slice)
    #        zPos (the z-position of the horizontal/x-axis slice)
    #Outputs: lineData (the plot data along the requested horizontal line)
    #         yPos_actual (the actual y-position of the horizontal slice/line)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      y-position that was requested by the user)
    #         zPos_actual (the actual z-position of the horizontal slice/line)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      z-position that was requested by the user)
    def __getXAxisSlice_3D(self, plotData, yCoords, zCoords, yPos, zPos):
        #Ensure that the simulation data is 3D
        assert(len(plotData.shape) == 3)
        #Get the y- and z-positions for the line
        if yCoords.shape[0] == 1:
            yPos_index = 0
            yPos_actual = yCoords[0,0,0]
        else:
            minY = yCoords.min()
            dy = yCoords[1,0,0] - yCoords[0,0,0]
            yPos_index = int((yPos - minY)/dy)
            assert (0 <= yPos_index < plotData.shape[0])
            yPos_actual = yCoords[yPos_index,0,0]
        if zCoords.shape[2] == 1:
            zPos_index = 0
            zPos_actual = zCoords[0,0,0]
        else:
            minZ = zCoords.min()
            dz = zCoords[0,0,1] - zCoords[0,0,0]
            zPos_index = int((zPos - minZ)/dz)
            assert (0 <= zPos_index < plotData.shape[2])
            zPos_actual = zCoords[0,0,zPos_index]
        #Retrieve the data along the line of cells with the y- and z-positions
        #that are closest to the values requested by the user
        lineData = plotData[yPos_index,:,zPos_index].reshape(-1)
        #Return the line of data, its actual y-position, and its actual z-position
        return lineData, yPos_actual, zPos_actual

    #Function: __getYAxisSlice_3D
    #Purpose: Retrieves the data along a particular line (parallel with the y-axis) in a 3D simulation.
    #Inputs: plotData (the 3D simulation data)
    #        xCoords (the x-coordinates of the cells in the simulation grid)
    #        zCoords (the z-coordinates of the cells in the simulation grid)
    #        xPos (the x-position of the vertical/y-axis slice)
    #        zPos (the z-position of the vertical/y-axis slice)
    #Outputs: lineData (the plot data along the requested vertical line)
    #         xPos_actual (the actual x-position of the vertical slice/line)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      y-position that was requested by the user)
    #         zPos_actual (the actual z-position of the vertical slice/line)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      z-position that was requested by the user)
    def __getYAxisSlice_3D(self, plotData, xCoords, zCoords, xPos, zPos):
        #Ensure that the simulation data is 3D
        assert(len(plotData.shape) == 3)
        #Get the x- and z-positions for the line
        if xCoords.shape[1] == 1:
            xPos_index = 0
            xPos_actual = xCoords[0,0,0]
        else:
            minX = xCoords.min()
            dx = xCoords[0,1,0] - xCoords[0,0,0]
            xPos_index = int((xPos - minX)/dx)
            assert (0 <= xPos_index < plotData.shape[1])
            xPos_actual = xCoords[0,xPos_index,0]
        if zCoords.shape[2] == 1:
            zPos_index = 0
            zPos_actual = zCoords[0,0,0]
        else:
            minZ = zCoords.min()
            dz = zCoords[0,0,1] - zCoords[0,0,0]
            zPos_index = int((zPos - minZ)/dz)
            assert (0 <= zPos_index < plotData.shape[2])
            zPos_actual = zCoords[0,0,zPos_index]
        #Retrieve the data along the line of cells with the x- and z-positions
        #that are closest to the values requested by the user
        lineData = plotData[:,xPos_index,zPos_index].reshape(-1)
        #Return the line of data, its actual x-position, and its actual z-position
        return lineData, xPos_actual, zPos_actual

    #Function: __getZAxisSlice_3D
    #Purpose: Retrieves the data along a particular line (parallel with the z-axis) in a 3D simulation.
    #Inputs: plotData (the 3D simulation data)
    #        xCoords (the x-coordinates of the cells in the simulation grid)
    #        yCoords (the y-coordinates of the cells in the simulation grid)
    #        xPos (the x-position of the depth/z-axis slice)
    #        yPos (the y-position of the depth/z-axis slice)
    #Outputs: lineData (the plot data along the requested line parallel with the z-axis)
    #         xPos_actual (the actual x-position of the z-axis slice/line)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      x-position that was requested by the user)
    #         yPos_actual (the actual y-position of the z-axis slice/line)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      y-position that was requested by the user)
    def getZAxisSlice_3D(self, plotData, xCoords, yCoords, xPos, yPos):
        #Ensure that the simulation data is 3D
        assert(len(plotData.shape) == 3)
        #Get the x- and y-positions for the line
        if xCoords.shape[2] == 1:
            xPos_index = 0
            xPos_actual = xCoords[0,0,0]
        else:
            minX = xCoords.min()
            dx = xCoords[0,1,0] - xCoords[0,0,0]
            xPos_index = int((xPos - minX)/dx)
            assert (0 <= xPos_index < plotData.shape[1])
            xPos_actual = xCoords[0,xPos_index,0]
        if yCoords.shape[1] == 1:
            yPos_index = 0
            yPos_actual = yCoords[0,0,0]
        else:
            minY = yCoords.min()
            dy = yCoords[1,0,0] - yCoords[0,0,0]
            yPos_index = int((yPos - minY)/dy)
            assert (0 <= yPos_index < plotData.shape[0])
            yPos_actual = yCoords[yPos_index,0,0]
        #Retrieve the data along the line of cells with the x- and z-positions
        #that are closest to the values requested by the user
        lineData = plotData[yPos_index,xPos_index,:].reshape(-1)
        #Return the line of data, its actual x-position, and its actual y-position
        return lineData, xPos_actual, yPos_actual

    #Function: __getXYPlaneSlice_3D
    #Purpose: Retrieves the data along a particular plane (parallel with the xy-plane) in a 3D simulation.
    #Inputs: plotData (the 3D simulation data)
    #        zCoords (the z-coordinates of the cells in the simulation grid)
    #        zPos (the z-position of the plane/cross-section)
    #Outputs: planeData (the plot data along the requested plane parallel with the xy-plane)
    #         zPos_actual (the actual z-position of the plane)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      z-position that was requested by the user)
    def __getXYPlaneSlice_3D(self, plotData, zCoords, zPos):
        assert(len(plotData.shape) == 3)
        #Find the z-position that is closest to the one requested by the user
        if zCoords.shape[2] == 1:
            zPos_index = 0
            zPos_actual = zCoords[0,0,0]
        else:
            minZ = zCoords.min()
            dz = zCoords[0,0,1] - zCoords[0,0,0]
            zPos_index = int((zPos - minZ)/dz)
            assert (0 < zPos_index < plotData.shape[2])
            zPos_actual = zCoords[0,0,zPos_index]
        #Retrieve the data within the requested plane
        planeData = plotData[:,:,zPos_index].reshape(plotData.shape[0],-1)
        #Return the plane of data and its actual z-position
        return planeData, zPos_actual

    #Function: __getXZPlaneSlice_3D
    #Purpose: Retrieves the data along a particular plane (parallel with the xz-plane) in a 3D simulation.
    #Inputs: plotData (the 3D simulation data)
    #        yCoords (the y-coordinates of the cells in the simulation grid)
    #        yPos (the y-position of the plane/cross-section)
    #Outputs: planeData (the plot data along the requested plane parallel with the xz-plane)
    #         zyPos_actual (the actual y-position of the plane)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      y-position that was requested by the user)
    def __getXZPlaneSlice_3D(self, plotData, yCoords, yPos):
        #Ensure that the simulation data is 3D
        assert(len(plotData.shape) == 3)
        #Get the y-position for the plane
        if yCoords.shape[0] == 1:
            yPos_index = 0
            yPos_actual = yCoords[0,0,0]
        else:
            minY = yCoords.min()
            dy = yCoords[1,0,0] - yCoords[0,0,0]
            yPos_index = int((yPos - minY)/dy)
            assert (0 <= yPos_index < plotData.shape[0])
            yPos_actual = yCoords[yPos_index,0,0]
        #Retrieve the data within the requested plane
        planeData = plotData[yPos_index,:,:].reshape(plotData.shape[1],-1)
        #Return the plane data and its actual y-position
        return planeData, yPos_actual

    #Function: __getYZPlaneSlice_3D
    #Purpose: Retrieves the data along a particular plane (parallel with the yz-plane) in a 3D simulation.
    #Inputs: plotData (the 3D simulation data)
    #        xCoords (the x-coordinates of the cells in the simulation grid)
    #        xPos (the x-position of the plane/cross-section)
    #Outputs: planeData (the plot data along the requested plane parallel with the yz-plane)
    #         xPos_actual (the actual x-position of the plane)
    #                     (necessary because there might be no cells that are centred on the exact
    #                      x-position that was requested by the user)
    def __getYZPlaneSlice_3D(self, plotData, xCoords, xPos):
        #Ensure that the simulation data is 3D
        assert(len(plotData.shape) == 3)
        #Get the x-position for the plane
        if xCoords.shape[1] == 1:
            xPos_index = 0
            xPos_actual = xCoords[0,0,0]
        else:
            minX = xCoords.min()
            dx = xCoords[0,1,0] - xCoords[0,0,0]
            xPos_index = int((xPos - minX)/dx)
            assert (0 <= xPos_index < plotData.shape[1])
            xPos_actual = xCoords[0,xPos_index,0]
        #Retrieve the data within the requested plane
        planeData = plotData[:,xPos_index,:].reshape(plotData.shape[0],-1)
        #Return the plane of data and its actual x-position
        return planeData, xPos_actual

    ######PUBLIC METHOD######
    #Function: visualize
    #Purpose: Updates all of the user's visual outputs for a simulation.
    #Inputs: simulationGrid (the Simulation Grid object that contains all of the new simulation data
    #                        we should use to update the visualizations)
    #        time (the current time in the simulation (so we can display the time in the titles of our visualizations)
    #Outputs: None (the changes will show up in the matplotlib figures)
    def visualize(self, simulationGrid, time):
        print("Visualizing Data")
        #Retrieve some parameter string constants that will help us efficiently
        #read information from our figure dictionaries
        figString = constants.FIGURE
        plotsString = constants.PLOTS
        colorbarsString = constants.COLORBARS
        axesString = constants.AXES
        numPlotsString = constants.NUM_PLOTS
        plotVarString = constants.PLOT_VAR
        plotTypeString = constants.PLOT_TYPE
        plotColorString = constants.PLOT_COLOR
        plotAlphaString = constants.PLOT_ALPHA
        minPlotValString = constants.MIN_PLOT_VAL
        hasMinPlotValString = constants.HAS_MIN_PLOT_VAL
        maxPlotValString = constants.MAX_PLOT_VAL
        hasMaxPlotValString = constants.HAS_MAX_PLOT_VAL
        plotAxisString = constants.PLOT_AXIS
        plotPlaneString = constants.PLOT_PLANE
        xPosString = constants.PLOT_X_POS
        yPosString = constants.PLOT_Y_POS
        zPosString = constants.PLOT_Z_POS
        xPosPrecString = constants.PLOT_X_POS_PREC
        yPosPrecString = constants.PLOT_Y_POS_PREC
        zPosPrecString = constants.PLOT_Z_POS_PREC
        #Iterate over all of the figures
        for figNum in range(self.numFigures):
            #Get the parameters for the current figure
            currFigurePar = self.figures[figNum]
            #Get the figure object
            currFigure = currFigurePar[figString]
            #Make this figure our current figure in matplotlib
            plt.figure(currFigure.number)
            #Remove the old colorbars (since they might have to change for the new data)
            for i in range(len(currFigurePar[colorbarsString])):
                if self.figures[figNum][colorbarsString][i] is not None:
                    self.figures[figNum][colorbarsString][i].ax.remove()
                self.figures[figNum][colorbarsString][i] = None
            #Iterate over the plots in the figure
            for plotNum in range(currFigurePar[numPlotsString]):
                #Get the parameters for the current plot
                currPlotPar = self.figures[figNum][plotsString][plotNum]
                #Get the string label for the plot's variable (i.e., the physical quantity it visualizes)
                plotVar = currPlotPar[plotVarString]
                #Get the new data for the plot
                plotData = self.__getPlotData(plotVar, simulationGrid)
                #Get the plot type and color
                plotType = currPlotPar[plotTypeString]
                plotColor = currPlotPar[plotColorString]
                #Get the axis object for the plot
                axis = currFigurePar[axesString][plotNum]
                #Check whether the plot is 1D, 2D or 3D
                if plotType == constants.PLOT_TYPE_1D:
                    #If the simulation grid is 1D, we can immediately update the 1D plot
                    if len(plotData.shape) == 1:
                        self.__update1DPlot(axis,simulationGrid.xCoords,plotVar,plotData,plotColor,constants.X_AXIS)
                        axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time))
                    elif len(plotData.shape) == 2:
                        #If the grid is 2D, we need to figure out which line through the simulation volume
                        #we are supposed to display.
                        axisType = currPlotPar[plotAxisString]
                        #Check if the line should be parallel with the x-axis or y-axis
                        if axisType == constants.X_AXIS:
                            #If the line is parallel with the x-axis, we need to know its y-position
                            yPos = currPlotPar[yPosString]
                            yPosPrec = currPlotPar[yPosPrecString]
                            #Retrieve the line of data from the simulation grid
                            (plotData, yPos_actual) = self.__getXAxisSlice_2D(plotData,simulationGrid.yCoords,yPos)
                            #Update the plot
                            self.__update1DPlot(axis,simulationGrid.xCoords[0,:].reshape(-1),plotVar,plotData,plotColor,
                                                constants.X_AXIS)
                            axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time)
                                           +("\n(Cross-Section at y = {yPosition:." + str(yPosPrec) + "f})").format(yPosition=yPos_actual))
                        else:
                            #If the line is parallel with the y-axis, we need to know its x-position
                            xPos = currPlotPar[xPosString]
                            xPosPrec = currPlotPar[xPosPrecString]
                            #Retrieve the line of data from the simulation grid
                            (plotData, xPos_actual) = self.__getYAxisSlice_2D(plotData,simulationGrid.xCoords,xPos)
                            #Update the plot
                            self.__update1DPlot(axis,simulationGrid.yCoords[:,0].reshape(-1),plotVar,plotData,plotColor,
                                                constants.Y_AXIS)
                            axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time)
                                           + ("\n(Cross-Section at x = {xPosition:." + str(xPosPrec) + "f})").format(xPosition=xPos_actual))
                    else: #If the grid is 3D
                        #Get the axis that should be parallel with the line through the 3D volume
                        axisType = currPlotPar[plotAxisString]
                        #Check which axis is parallel with the line
                        if axisType == constants.X_AXIS:
                            #If the line is parallel with the x-axis, we need to know its y- and z-position
                            yPos = currPlotPar[yPosString]
                            yPosPrec = currPlotPar[yPosPrecString]
                            zPos = currPlotPar[zPosString]
                            zPosPrec = currPlotPar[zPosPrecString]
                            #Retrieve the line of data from the simulation grid
                            (plotData, yPos_actual, zPos_actual) = self.__getXAxisSlice_3D(plotData,simulationGrid.yCoords,
                                                                                         simulationGrid.zCoords,yPos,zPos)
                            #Update the plot
                            self.__update1DPlot(axis,simulationGrid.xCoords[0,:,0].reshape(-1),plotVar,plotData,
                                                plotColor,constants.X_AXIS)
                            axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time)
                            +("\n(Cross-Section at y = {yPosition:." + str(yPosPrec)
                                   + "f} and z = {yPosition:." + str(zPosPrec) + "f})").format(yPosition=yPos_actual,
                                                                                                       zPosition=zPos_actual))
                        elif axisType == constants.Y_AXIS:
                            #If the line is parallel with the y-axis, we need to know its x- and z-position
                            xPos = currPlotPar[xPosString]
                            xPosPrec = currPlotPar[xPosPrecString]
                            zPos = currPlotPar[zPosString]
                            zPosPrec = currPlotPar[zPosPrecString]
                            #Retrieve the line of data from the simulation grid
                            (plotData, xPos_actual, zPos_actual) = self.__getYAxisSlice_3D(plotData,simulationGrid.xCoords,
                                                                                           simulationGrid.zCoords,xPos,zPos)
                            #Update the plot
                            self.__update1DPlot(axis,simulationGrid.yCoords[:,0,0].reshape(-1),plotVar,plotData,plotColor,
                                                constants.Y_AXIS)
                            axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time)
                            +("\n(Cross-Section at x = {xPosition:." + str(xPosPrec)
                                   + "f} and z = {zPosition:." + str(zPosPrec) + "f})").format(xPosition=xPos_actual,
                                                                                                       zPosition=zPos_actual))
                        else: #if the line is parallel with the Z_AXIS
                            #If the line is parallel with the z-axis, we need to know its x- and y-position
                            xPos = currPlotPar[xPosString]
                            xPosPrec = currPlotPar[xPosPrecString]
                            yPos = currPlotPar[yPosString]
                            yPosPrec = currPlotPar[yPosPrecString]
                            #Retrieve the line of data from the simulation grid
                            (plotData, xPos_actual, yPos_actual) = self.getZAxisSlice_3D(plotData,simulationGrid.xCoords,
                                                                                         simulationGrid.yCoords,xPos,yPos)
                            #Update the plot
                            self.__update1DPlot(axis,simulationGrid.zCoords[0,0,:].reshape(-1),plotVar,plotData,plotColor,
                                                constants.Z_AXIS)
                            axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time)
                            +("\n(Cross-Section at x = {xPosition:." + str(xPosPrec)
                                   + "f} and y = {yPosition:." + str(yPosPrec) + "f})").format(xPosition=xPos_actual,
                                                                                                yPosition=yPos_actual))
                elif plotType == constants.PLOT_TYPE_2D:
                    #If the plot type is 2D, we need to check whether the simulation is 2D or 3D
                    if len(plotData.shape) == 2: #If the simulation is 2D
                        #Check whether the user has set min and/or max values for the colormap
                        hasMinPlotVal = currPlotPar[hasMinPlotValString]
                        minPlotVal = plotData.min()
                        if hasMinPlotVal:
                            minPlotVal = currPlotPar[minPlotValString]
                        hasMaxPlotVal = currPlotPar[hasMaxPlotValString]
                        maxPlotVal = plotData.max()
                        if hasMaxPlotVal:
                            maxPlotVal = currPlotPar[maxPlotValString]
                        #Update the 2D colormap
                        self.__update2DPlot(figNum,plotNum,axis,simulationGrid,plotData,plotColor,constants.X_AXIS,
                                            constants.Y_AXIS,minPlotVal,maxPlotVal)
                        axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time))
                    else: #If the simulation is 3D
                        #Find the plane that our 2D cross-section should be parallel with
                        planeType = currPlotPar[plotPlaneString]
                        if planeType == constants.XY_PLANE:
                            #If the cross-section should be parallel with the xy-plane, we need to know its z-position
                            zPos = currPlotPar[zPosString]
                            zPosPrec = currPlotPar[zPosPrecString]
                            #Get the data for the cross-section
                            (plotData, zPos_actual) = self.__getXYPlaneSlice_3D(plotData,simulationGrid.zCoords,zPos)
                            #Check if the user has provided min and/or max values for the colormap
                            hasMinPlotVal = currPlotPar[hasMinPlotValString]
                            minPlotVal = plotData.min()
                            if hasMinPlotVal:
                                minPlotVal = currPlotPar[minPlotValString]
                            hasMaxPlotVal = currPlotPar[hasMaxPlotValString]
                            maxPlotVal = plotData.max()
                            if hasMaxPlotVal:
                                maxPlotVal = currPlotPar[maxPlotValString]
                            #Update the 2D colormap
                            self.__update2DPlot(figNum,plotNum,axis,simulationGrid,plotData,plotColor,constants.X_AXIS,
                                                constants.Y_AXIS,minPlotVal,maxPlotVal)
                            axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time)
                                            + ("\n(Cross-Section at z = {zPosition:." + str(zPosPrec) + "f})").format(zPosition=zPos_actual))
                        elif planeType == constants.XZ_PLANE:
                            #If the cross-section should be parallel with the xz-plane, we need to know its y-position
                            yPos = currPlotPar[yPosString]
                            yPosPrec = currPlotPar[yPosPrecString]
                            #Get the data for the cross-section
                            (plotData, yPos_actual) = self.__getXZPlaneSlice_3D(plotData,simulationGrid.yCoords,yPos)
                            #Check if the user has provided min and/or max values for the colormap
                            hasMinPlotVal = currPlotPar[hasMinPlotValString]
                            minPlotVal = plotData.min()
                            if hasMinPlotVal:
                                minPlotVal = currPlotPar[minPlotValString]
                            hasMaxPlotVal = currPlotPar[hasMaxPlotValString]
                            maxPlotVal = plotData.max()
                            if hasMaxPlotVal:
                                maxPlotVal = currPlotPar[maxPlotValString]
                            #Update the 2D colormap
                            self.__update2DPlot(figNum,plotNum,axis,simulationGrid,plotData,plotColor,constants.X_AXIS,
                                                constants.Z_AXIS,minPlotVal,maxPlotVal)
                            axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time)
                                            + ("\n(Cross-Section at y = {yPosition:.)" + str(yPosPrec) + "f})").format(yPosition=yPos_actual))
                        else:
                            #If the cross-section should be parallel with the yz-plane, we need to know its x-position
                            xPos = currPlotPar[xPosString]
                            xPosPrec = currPlotPar[xPosPrecString]
                            #Get the data for the cross-section
                            (plotData, xPos_actual) = self.__getYZPlaneSlice_3D(plotData,simulationGrid.xCoords,xPos)
                            #Check if the user has provided min and/or max values for the colormap
                            hasMinPlotVal = currPlotPar[hasMinPlotValString]
                            minPlotVal = plotData.max()
                            if hasMinPlotVal:
                                minPlotVal = currPlotPar[minPlotValString]
                            hasMaxPlotVal = currPlotPar[hasMaxPlotValString]
                            maxPlotVal = plotData.min()
                            if hasMaxPlotVal:
                                maxPlotVal = currPlotPar[maxPlotValString]
                            #Update the 2D colormap
                            self.__update2DPlot(figNum, plotNum,axis,simulationGrid,plotData,plotColor,constants.Y_AXIS,
                                                constants.Z_AXIS,minPlotVal,maxPlotVal)
                            axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time)
                                            + ("\n(Cross-Section at x = {xPosition:.)" + str(xPosPrec) + "f})").format(xPosition=xPos_actual))
                else: #If the plot type is 3D
                    #Get the transparency/alpha value for the 3D plot
                    plotAlpha = currPlotPar[plotAlphaString]
                    #Update the plot
                    self.__update3DPlot(figNum, plotNum, axis,simulationGrid,plotData,plotColor,plotAlpha,
                                      currPlotPar[hasMinPlotValString],currPlotPar[minPlotValString],
                                      currPlotPar[hasMaxPlotValString],currPlotPar[maxPlotValString])
                    axis.set_title(plotVar + (" at Time t = {time:." + str(self.timePrec) + "f}").format(time=time))
            #Display/draw all of the changes we made to the plots on the figure
            plt.draw()
            plt.pause(0.1)


    def blockOnPlot(self):
        plt.show()





        # if self.numPlots == 1:
        #     plotData = self.getPlotData(self.plotVar, simulationGrid)
        #     plt.cla()
        #     if self.plotType == constants.PLOT_TYPE_1D:
        #         self.ax.plot(simulationGrid.xCoords,plotData,color=self.plotColor)
        #         self.ax.set_xlim(simulationGrid.minX, simulationGrid.maxX)
        #         self.ax.set_xlabel("x-position")
        #         self.ax.set_ylabel(self.plotVar)
        #     else:
        #         minPlotVal = plotData.min()
        #         maxPlotVal = plotData.max()
        #         if self.hasMinPlotVal:
        #             minPlotVal = self.minPlotVal
        #         if self.hasMaxPlotVal:
        #             maxPlotVal = self.maxPlotVal
        #         if self.plotType == constants.PLOT_TYPE_2D:
        #             img = plt.imshow(plotData,
        #                              extent=[simulationGrid.minX, simulationGrid.maxX,
        #                                      simulationGrid.minY, simulationGrid.maxY],
        #                              cmap=self.plotColor, vmin=minPlotVal, vmax=maxPlotVal)
        #         else:
        #             img = self.ax.scatter3D(simulationGrid.xCoords, simulationGrid.yCoords, simulationGrid.zCoords,
        #                                     alpha=self.plotAlpha, c=plotData.reshape(-1), cmap=self.plotColor,
        #                                     vmin=minPlotVal, vmax=maxPlotVal)
        #             self.ax.set_zlabel("z-position")
        #         if time == 0.0:
        #             self.colorbar = self.fig.colorbar(img)
        #             self.colorbar.set_alpha(1)
        #             self.colorbar.draw_all()
        #         self.ax.set_xlabel("x-position")
        #         self.ax.set_ylabel("y-position")
        #     self.ax.set_title(self.plotVar + " at Time t = {time:.3f}".format(time = time))
        #     plt.draw()
        #     plt.pause(0.1)
        #     if time == tLim:
        #         plt.show()






