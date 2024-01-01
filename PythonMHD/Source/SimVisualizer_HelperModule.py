#SimVisualizer_HelperModule.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides the supporting functions for allowing the SimVisualizer class to validate
#         all of the user's visualization settings/parameters.

######IMPORT STATEMENTS######

#Import the matplotlib function for checking whether a variable can be interpreted as a color
from matplotlib.colors import is_color_like

#Import the set of colormaps that matplotlib supports
from matplotlib.pyplot import colormaps

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants


######PLOT VARIABLE VALIDATORS######

#Function: checkHydroPlotVar
#Purpose: Checks whether the user has requested a valid hydrodynamic variable.
#Inputs: plotVar (the string for the variable that the user has requested)
#        figNum (the figure number (so we can tell the user which figure is causing an error, if there is one))
#        plotNum (the plot number (so we can tell the user which plot is causing an error, if there is one))
#Outputs: None (if validation is successful)
#         RuntimeError if the requested variable is invalid
def checkHydroPlotVar(plotVar, figNum, plotNum):
    if plotVar not in constants.HYDRO_PLOT_VARS:
        raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum) + "):\n"
                           +"The plotVar value (plotVar = " + plotVar + ") in the visualization parameter "
                           + "structure is invalid.\nThe valid plotVar inputs for a hydrodynamic "
                           + "simulation are listed below:\n " + str(constants.HYDRO_PLOT_VARS))

#Function: checkMHDPlotVar
#Purpose: Checks whether the user has requested a valid MHD variable.
#Inputs: plotVar(the string for the variable that the user has requested)
#Outputs: None (if validation is successful)
#         RuntimeError if the requested variable is invalid
def checkMHDPlotVar(plotVar, figNum, plotNum):
    if plotVar not in constants.MHD_PLOT_VARS:
        raise RuntimeError( "VISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum) + "):\n"
                           +"The plotVar value (plotVar = " + plotVar + ") in the visualization "
                           + "parameter structure is invalid. The valid plotVar inputs for a hydrodynamic "
                                       + "simulation are listed below:\n " + str(constants.MHD_PLOT_VARS))

######PLOT TYPE VALIDATOR######
#Function: checkPlotType
#Purpose: Checks whether the user has requested a valid plot type (1D, 2D or 3D) for their simulation.
#Inputs: plotPar (the user's parameters dictionary for the plot)
#        simulationGrid (the SimulationGrid object for the simulation at the current time)
#Outputs: None (if validation is successful)
#         RuntimeError if the plot type is invalid
def checkPlotType(plotPar, simulationGrid, figNum, plotNum):
    #Retrieve parameter string constants that will help us read the visPar dictionary
    plotTypeString = constants.PLOT_TYPE
    axisString = constants.PLOT_AXIS
    planeString = constants.PLOT_PLANE
    xPosString = constants.PLOT_X_POS
    yPosString = constants.PLOT_Y_POS
    zPosString = constants.PLOT_Z_POS
    #Get the number of spatial dimensions (1, 2, or 3)
    nDim = simulationGrid.nDim
    #Ensure that the simulation is 1D, 2D, or 3D
    assert nDim == 1 or nDim == 2 or nDim == 3
    #We only need to validate the plot type if the user has overriden the default value
    #by setting their own "plotType" parameter in visPar
    if plotTypeString in plotPar.keys():
        #Get the user's plot type from visPar
        plotType = plotPar[plotTypeString]
        #Raise an error if the plot type is not a string
        if plotType not in constants.VALID_PLOT_TYPES:
            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                               + "You have passed an invalid data type for " + constants.PLOT_TYPE + " in your "
                               + "visualization parameters structure.\nThe " + constants.PLOT_TYPE + " parameter must be a string that "
                               + "corresponds to a visualization type that is supported by PythonMHD.\nValid plotType inputs for a PythonMHD "
                               + "simulation are listed below:\n" + str(constants.VALID_PLOT_TYPES))
        #Raise an error if the plot type is not supported by PythonMHD
        elif plotType not in constants.VALID_PLOT_TYPES:
            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                               + "You have requested a plot type (plotType = " + plotType
                               + ") in your visualization parameters structure that is not "
                               + "supported by PythonMHD.\nValid plotType inputs for a PythonMHD "
                               + "simulation are listed below:\n" + str(constants.VALID_PLOT_TYPES))
        #How we validate the plot type depends on whether the simulation is 1D, 2D, or 3D
        if nDim == 1:
            #For 1D sims, a 1D line plot is the only option.
            if plotType != constants.PLOT_TYPE_1D:
                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                   + "You have submitted an invalid plot type (plotType = " + plotType + ") "
                                   + "for a 1D simulation.\nThe only supported plot type for a 1D "
                                   + "simulation is \"" + constants.PLOT_TYPE_1D + "\".")
        if nDim == 2:
            #For 2D sims, the user can choose between a 2D colormap and a 1D line plot.
            if plotType != constants.PLOT_TYPE_1D and plotType != constants.PLOT_TYPE_2D:
                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                   + "You have submitted an invalid plot type (plotType = " + plotType + ") "
                                   + "for a 2D simulation.\nThe only supported plot types for a 2D "
                                   + "simulation are \"" + constants.PLOT_TYPE_1D + "\" and \"" + constants.PLOT_TYPE_2D + "\".")
            #For 2D sims, the user can choose between full 2D colormap of their data or a 1D line plot of any horizontal
            #or vertical slice through the simulation grid.
            if plotType == constants.PLOT_TYPE_1D: #if the user wants a line plot
                #Make sure the user has specified which axis the line should be parallel with
                if axisString not in plotPar.keys():
                    raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                       + "If you want to plot the 1D cross-section of a 2D simulation,\n"
                                       + "you must specify the axis of the cross-section in your "
                                       + "visualization parameters structure.\n\nIf you want to plot "
                                       + "a slice along the x-axis (with constant y),\nset \"" + axisString
                                       + "\" = \"" + constants.X_AXIS + "\".\n\nIf you want to plot a slice along the "
                                       + "the y-axis (with constant x),\nset \"" + axisString + "\" = \""
                                       + constants.Y_AXIS + "\".")
                else: #if the user has specifed the plot axis
                    #Make sure the axis is valid (x, y, or z)
                    if plotPar[axisString] not in constants.VALID_3D_AXES:
                        raise RuntimeError("\nVISUALIZATION ERROR (Figure " + str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                           + "\nYou passed an invalid value for " + axisString + " "
                                           + "in your visualization parameters structure.\nThe valid "
                                           + "axis values for a 2D cartesian simulation are listed "
                                           + "below:\n" + str(constants.VALID_2D_AXES))
                    elif plotPar[axisString] == constants.X_AXIS:
                        #If the line should be parallel with the x-axis, the user also needs to specify the
                        #y-position of the line.
                        if yPosString not in plotPar.keys():
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You did not specify the y-position of the x-axis cross-section "
                                               + "that you want to plot for the 2D simulation.\nPlease "
                                               + "include the parameter " + yPosString + " in your "
                                               + "visualization parameters structure.")
                        elif type(plotPar[yPosString]) != int and type(plotPar[yPosString]) != float:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure " + str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You passed an invalid value for " + yPosString + " "
                                               + "in your visualization parameters structure.\nThe y-position of your "
                                               + "x-axis cross-section must be a numeric value (i.e., an integer or a "
                                               + "float).")
                        else: #if the user has provided the y-position of the line
                            #Make sure the y-position is a valid coordinate in the simulation grid
                            yPos = plotPar[yPosString]
                            if yPos < simulationGrid.minY or yPos > simulationGrid.maxY:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid value for " + yPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "y-position that you passed (" + yPosString + " = "
                                                   + str(yPos) + ") does not fall within the min and max "
                                                   + "y-coordinates\non the simulation grid (minY = "
                                                   + str(simulationGrid.minY) + ", maxY = "
                                                   + str(simulationGrid.maxY) + ").")
                    else: #if the line should be parallel with the y-axis
                        #If the line should be parallel with the y-axis, the user also needs to specify the
                        #x-position of the line.
                        if xPosString not in plotPar.keys():
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum) + "):\n"
                                               + "You did not specify the x-position of the y-axis cross-section "
                                               + "that you want to plot for the 2D simulation.\nPlease "
                                               + "include the parameter " + xPosString + " in your "
                                               + "visualization parameters structure.")
                        elif type(plotPar[xPosString]) != int and type(plotPar[xPosString]) != float:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure " + str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You passed an invalid value for " + xPosString + " "
                                               + "in your visualization parameters structure.\nThe x-position of your "
                                               + "y-axis cross-section must be a numeric value (i.e., an integer or a "
                                               + "float).")
                        else:
                            #Make sure the x-position is a valid coordinate in the simulation grid
                            xPos = plotPar[xPosString]
                            if xPos < simulationGrid.minX or xPos > simulationGrid.maxX:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid value for " + xPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "x-position that you passed (" + xPosString + " = "
                                                   + str(xPos) + ") does not fall within the min and max "
                                                   + "x-coordinates\non the simulation grid (minX = "
                                                   + str(simulationGrid.minX) + ", maxX = "
                                                   + str(simulationGrid.maxX) + ").")
        if nDim == 3: #if the simulation is 3D
            #For 3D simulations, users can choose between a 3D transparent colormap of the entire simulation volume,
            #a 2D colormap of any plane (parallel with the xy-plane, yz-plane, or xz-plane) in the simulation grid,
            #or a 1D line plot of any line through the simulation that is parallel with the x-, y-, or z-axis.
            if plotType == constants.PLOT_TYPE_1D:
                #If the user wants a 1D line plot, they need to specify whether it should be parallel
                #with the x-, y-, or z-axis.
                if axisString not in plotPar.keys():
                    raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                       + "If you want to plot the 1D cross-section of a 3D simulation, "
                                       + "you must specify the axis of the cross-section\nin your "
                                       + "visualization parameters structure.\n\nIf you want to plot "
                                       + "a slice along the x-axis (with constant y and z), set \"" + axisString
                                       + "\" = \"" + constants.X_AXIS + "\".\n\nIf you want to plot a slice along the "
                                       + "y-axis (with constant x and z), set \"" + axisString + "\" = \""
                                       + constants.Y_AXIS + "\".\n\nIf you want to plot a slice along the x-axis "
                                       + "(with constant x and y), set \"" + axisString + "\" = \""
                                       + constants.X_AXIS + "\".")
                else: #if the user has specified the axis for the line
                    #Make sure that the axis is valid for a 3D sim
                    if plotPar[axisString] not in constants.VALID_3D_AXES:
                        raise RuntimeError("VISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                           + "You passed an invalid value for " + axisString + " "
                                           + "in your visualization parameters structure. The valid "
                                           + "axis values for a 3D cartesian simulation are listed "
                                           + "below: " + str(constants.VALID_3D_AXES))
                    elif plotPar[axisString] == constants.X_AXIS:
                        #If the line is parallel with the x-axis, the user needs to specify its y- and z-positions.
                        if yPosString not in plotPar.keys():
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You did not specify the y-position of the x-axis cross-section "
                                               + "that you want to plot for the 3D simulation.\nPlease "
                                               + "include the parameter " + yPosString + " in your "
                                               + "visualization parameters structure.")
                        elif zPosString not in plotPar.keys():
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You did not specify the z-position of the x-axis cross-section "
                                               + "that you want to plot for the 3D simulation.\nPlease "
                                               + "include the parameter " + zPosString + " in your "
                                               + "visualization parameters structure.")
                        else: #If the user has provided the y- and z-positions of the line
                            #Make sure that the y- and z-positions are valid
                            yPos = plotPar[yPosString]
                            zPos = plotPar[zPosString]
                            #Make sure the y-position is a numeric value
                            if type(yPos) != int and type(yPos) != float:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid data type for the y-position of the x-axis line.\n"
                                                   + "You need to pass a numeric value (i.e., an int or a float) for the "
                                                   + yPosString + " parameter in your visPar structure.")
                            #Make sure the z-position is a numeric value
                            elif type(zPos) != int and type(zPos) != float:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid data type for the z-position of the x-axis line.\n"
                                                   + "You need to pass a numeric value (i.e., an int or a float) for the "
                                                   + zPosString + " parameter in your visPar structure.")
                            if yPos < simulationGrid.minY or yPos > simulationGrid.maxY:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid value for " + yPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "y-position that you passed (" + yPosString + " = "
                                                   + str(yPos) + ") does not fall within the min and max "
                                                   + "y-coordinates\non the simulation grid (minY = "
                                                   + str(simulationGrid.minY) + ", maxY = "
                                                   + str(simulationGrid.maxY) + ").")
                            elif zPos < simulationGrid.minZ or zPos > simulationGrid.maxZ:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid value for " + zPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "z-position that you passed (" + zPosString + " = "
                                                   + str(zPos) + ") does not fall within the min and max "
                                                   + "z-coordinates\non the simulation grid (minZ = "
                                                   + str(simulationGrid.minZ) + ", maxZ = "
                                                   + str(simulationGrid.maxZ) + ").")
                    elif plotPar[axisString] == constants.Y_AXIS:
                        #If the line is parallel with the y-axis, the user needs to specify its x- and z-positions.
                        if xPosString not in plotPar.keys():
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You did not specify the x-position of the y-axis cross-section "
                                               + "that you want to plot for the 3D simulation.\nPlease "
                                               + "include the parameter " + xPosString + " in your "
                                               + "visualization parameters structure.")
                        elif zPosString not in plotPar.keys():
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You did not specify the z-position of the y-axis cross-section "
                                               + "that you want to plot for the 3D simulation.\nPlease "
                                               + "include the parameter " + zPosString + " in your "
                                               + "visualization parameters structure.")
                        else: #If the user has provided the x- and z-positions of the line
                            #Make sure that the x- and z-positions are valid
                            xPos = plotPar[xPosString]
                            zPos = plotPar[zPosString]
                            #Make sure the x-position is a numeric value
                            if type(xPos) != int and type(xPos) != float:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid data type for the x-position of the y-axis line.\n"
                                                   + "You need to pass a numeric value (i.e., an int or a float) for the "
                                                   + xPosString + " parameter in your visPar structure.")
                            #Make sure the z-position is a numeric value
                            elif type(zPos) != int and type(zPos) != float:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid data type for the z-position of the y-axis line.\n"
                                                   + "You need to pass a numeric value (i.e., an int or a float) for the "
                                                   + zPosString + " parameter in your visPar structure.")
                            if xPos < simulationGrid.minX or xPos > simulationGrid.maxX:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid value for " + xPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "x-position that you passed (" + xPosString + " = "
                                                   + str(xPos) + ") does not fall within the min and max "
                                                   + "x-coordinates\non the simulation grid (minX = "
                                                   + str(simulationGrid.minX) + ", maxX = "
                                                   + str(simulationGrid.maxX) + ").")
                            elif zPos < simulationGrid.minZ or zPos > simulationGrid.maxZ:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid value for " + zPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "z-position that you passed (" + zPosString + " = "
                                                   + str(zPos) + ") does not fall within the min and max "
                                                   + "z-coordinates\non the simulation grid (minZ = "
                                                   + str(simulationGrid.minZ) + ", maxZ = "
                                                   + str(simulationGrid.maxZ) + ").")
                    else: #if the line is parallel with the z-axis
                        #If the line is parallel with the z-axis, the user needs to specify its x- and y-positions.
                        if xPosString not in plotPar.keys():
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You did not specify the x-position of the z-axis cross-section "
                                               + "that you want to plot for the 3D simulation.\nPlease "
                                               + "include the parameter " + xPosString + " in your "
                                               + "visualization parameters structure.")
                        elif yPosString not in plotPar.keys():
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You did not specify the y-position of the z-axis cross-section "
                                               + "that you want to plot for the 3D simulation.\nPlease "
                                               + "include the parameter " + yPosString + " in your "
                                               + "visualization parameters structure.")
                        else: #If the user has provided the x- and y-positions of the line
                            #Make sure that the x- and y-positions are valid
                            xPos = plotPar[xPosString]
                            yPos = plotPar[yPosString]
                            #Make sure the x-position is a numeric value
                            if type(xPos) != int and type(xPos) != float:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid data type for the x-position of the z-axis line.\n"
                                                   + "You need to pass a numeric value (i.e., an int or a float) for the "
                                                   + zPosString + " parameter in your visPar structure.")
                            #Make sure the y-position is a numeric value
                            elif type(yPos) != int and type(yPos) != float:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid data type for the y-position of the z-axis line.\n"
                                                   + "You need to pass a numeric value (i.e., an int or a float) for the "
                                                   + yPosString + " parameter in your visPar structure.")
                            elif xPos < simulationGrid.minX or xPos > simulationGrid.maxX:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid value for " + xPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "x-position that you passed (" + xPosString + " = "
                                                   + str(xPos) + ") does not fall within the min and max "
                                                   + "x-coordinates\non the simulation grid (minX = "
                                                   + str(simulationGrid.minX) + ", maxX = "
                                                   + str(simulationGrid.maxX) + ").")
                            elif yPos < simulationGrid.minY or yPos > simulationGrid.maxY:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid value for " + yPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "y-position that you passed (" + yPosString + " = "
                                                   + str(yPos) + ") does not fall within the min and max "
                                                   + "y-coordinates\non the simulation grid (minY = "
                                                   + str(simulationGrid.minY) + ", maxY = "
                                                   + str(simulationGrid.maxY) + ").")
            elif plotType == constants.PLOT_TYPE_2D:
                #If the user wants a 2D plot of their 3D simulation, they need to specify whether the
                #2D cross-section should be parallel with the xy-, yz-, or xz-plane.
                if planeString not in plotPar.keys():
                    raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                       + "If you want to plot the 2D cross-section of a 3D simulation, "
                                       + "\nyou must specify the plane of the cross-section in your "
                                       + "visualization parameters structure.\n\nIf you want to plot "
                                       + "a slice along the xy-plane (with constant z), set \"" + planeString
                                       + "\" = \"" + constants.XY_PLANE + "\".\n\nIf you want to plot a slice along the "
                                       + "xz-plane (with constant y), set \"" + planeString + "\" = \""
                                       + constants.XZ_PLANE + "\".\n\nIf you want to plot a slice along the yz-plane "
                                       + "(with constant x), set \"" + planeString + "\" = \"" + constants.YZ_PLANE
                                       + "\".")
                else:
                    #Make sure that the user has requested a valid plane
                    if plotPar[planeString] not in constants.VALID_3D_PLANES:
                        raise RuntimeError("VISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum) + "):\n"
                                           + "You passed an invalid value for " + planeString + " "
                                           + "in your visualization parameters structure. The valid "
                                           + "plane values for a 3D cartesian simulation are listed "
                                           + "below: " + str(constants.VALID_3D_PLANES))
                    elif plotPar[planeString] == constants.XY_PLANE:
                        #If the cross-section should be parallel with the xy-plane, we need to know its z-position.
                        if zPosString not in plotPar.keys():
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You did not specify the z-position of the xy-plane "
                                               + "that you want to plot for the 3D simulation.\nPlease "
                                               + "include the parameter " + zPosString + " in your "
                                               + "visualization parameters structure.")
                        #Make sure the z-position is a numeric value
                        elif type(plotPar[zPosString]) != int and type(plotPar[zPosString]) != float:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You passed an invalid data type for the z-position of the xy-plane.\n"
                                               + "You need to pass a numeric value (i.e., an int or a float) for the "
                                               + zPosString + " parameter in your visPar structure.")
                        else: #if the user has provided the z-position for the 2D plot
                            #Make sure that the z-position is a valid coordinate in the simulation
                            zPos = plotPar[zPosString]
                            if zPos < simulationGrid.minZ or zPos > simulationGrid.maxZ:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid value for " + zPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "z-position that you passed (" + zPosString + " = "
                                                   + str(zPos) + ") does not fall within the min and max "
                                                   + "z-coordinates\non the simulation grid (minZ = "
                                                   + str(simulationGrid.minZ) + ", maxZ = "
                                                   + str(simulationGrid.maxZ) + ").")
                    elif plotPar[planeString] == constants.XZ_PLANE:
                        #If the cross-section should be parallel with the xz-plane, we need to know its y-position.
                        if yPosString not in plotPar.keys():
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You did not specify the y-position of the xz-plane "
                                               + "that you want to plot for the 3D simulation.\nPlease "
                                               + "include the parameter " + yPosString + " in your "
                                               + "visualization parameters structure.")
                        #Make sure the y-position is a numeric value
                        elif type(plotPar[yPosString]) != int and type(plotPar[yPosString]) != float:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You passed an invalid data type for the y-position of the xz-plane.\n"
                                               + "You need to pass a numeric value (i.e., an int or a float) for the "
                                               + yPosString + " parameter in your visPar structure.")
                        else: #if the user has provided the y-position for the 2D plot
                            #Make sure that the y-position is a valid coordinate in the simulation
                            yPos = plotPar[yPosString]
                            if yPos < simulationGrid.minY or yPos > simulationGrid.maxY:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "\nYou passed an invalid value for " + yPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "y-position that you passed (" + yPosString + " = "
                                                   + str(yPos) + ") does not fall within the min and max "
                                                   + "y-coordinates\non the simulation grid (minY = "
                                                   + str(simulationGrid.minY) + ", maxY = "
                                                   + str(simulationGrid.maxY) + ").")
                    else: #if the plot should be parallel with the yz-plane
                        #If the cross-section should be parallel with the yz-plane, we need to know its x-position.
                        if xPosString not in plotPar.keys():
                            raise RuntimeError("V\nISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You did not specify the x-position of the yz-plane "
                                               + "that you want to plot for the 3D simulation.\nPlease "
                                               + "include the parameter " + xPosString + " in your "
                                               + "visualization parameters structure.")
                        #Make sure the x-position is a numeric value
                        elif type(plotPar[xPosString]) != int and type(plotPar[xPosString]) != float:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                               + "You passed an invalid data type for the x-position of the yz-plane.\n"
                                               + "You need to pass a numeric value (i.e., an int or a float) for the "
                                               + xPosString + " parameter in your visPar structure.")
                        else: #if the user has provided the x-position for the 2D plot
                            #Make sure that the x-position is a valid coordinate in the simulation
                            xPos = plotPar[xPosString]
                            if xPos < simulationGrid.minX or xPos > simulationGrid.maxX:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                                   + "You passed an invalid value for " + xPosString + " "
                                                   + "in your visualization parameters structure.\nThe "
                                                   + "x-position that you passed (" + xPosString + " = "
                                                   + str(xPos) + ") does not fall within the min and max "
                                                   + "x-coordinates\non the simulation grid (minX = "
                                                   + str(simulationGrid.minX) + ", maxX = "
                                                   + str(simulationGrid.maxX) + ").")

######PLOT COLOR VALIDATOR######
#Function: checkPlotColor
#Purpose: Checks whether the user has requested a valid matplotlib color for their data plot.
#         For 3D visualizations, it also checks whether the alpha/transparency value is valid.
#Inputs: plotPar (the user's parameters dictionary for the plot)
#        nDim (the number of spatial dimensions in the simulation (1 for 1D, 2 for 2D, and 3 for 3D))
#        figNum (the figure number (so we can tell the user which figure is causing an error, if there is one))
#        plotNum (the plot number (so we can tell the user which plot is causing an error, if there is one))
#Outputs: None (if validation is successful)
#         RuntimeError if the color or alpha value is invalid
def checkPlotColor(plotPar, nDim, figNum, plotNum):
    #Get some parameter string constants that will help us efficiently read from plotPar
    plotColorString = constants.PLOT_COLOR
    plotTypeString = constants.PLOT_TYPE
    plotAlphaString = constants.PLOT_ALPHA
    minPlotValString = constants.MIN_PLOT_VAL
    maxPlotValString = constants.MAX_PLOT_VAL
    #Ensure that the simulation is 1D, 2D, or 3D
    assert nDim == 1 or nDim == 2 or nDim == 3, "checkPlotColor in SimVisualizer_HelperModule.py encountered a simulation "\
                                                + "with an invalid number of spatial dimensions"
    #If the user has not specified the plot type, we assume that they want the default plot type for the number of
    #spatial dimensions in their simulation.
    if plotTypeString not in plotPar.keys():
        if nDim == 1:
            plotType = constants.PLOT_TYPE_1D
        elif nDim == 2:
            plotType = constants.PLOT_TYPE_2D
        else:
            plotType = constants.PLOT_TYPE_3D
    else: #If the user has specified their own plot type
        #Double-check that the plot type is valid
        assert plotPar[plotTypeString] in constants.VALID_PLOT_TYPES, "checkPlotColor in SimVisualizer_HelperModule.py " \
                                                                      + "encountered an invalid plot type (figure "\
                                                                      + str(figNum) + ", plot " + str(plotNum) + ")"
        plotType = plotPar[plotTypeString]
    #Check if the user has specified a color for the plot
    if plotColorString in plotPar.keys():
        #For 1D, we check if the plot color can be read as a valid matplotlib color.
        if plotType == constants.PLOT_TYPE_1D:
            if not is_color_like(plotPar[plotColorString]):
                raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                   + "You passed an invalid value for " + plotColorString + ". "
                                   + "PythonMHD will accept any color that is supported by "
                                   + "matplotlib.\nSee https://matplotlib.org/stable/tutorials/colors/colors.html "
                                   + "for details on how you can specify colors for matplotlib.")
        else: #For 2D and 3D, we need to check if the user has chosen a valid matplotlib colormap.
            if plotPar[plotColorString] not in colormaps():
                raise RuntimeError("VISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                   + "You passed an invalid value for " + plotColorString + " "
                                   + "PythonMHD will accept any colormap that is supported by "
                                   + " matplotlib. See https://matplotlib.org/stable/tutorials/colors/colors.html"
                                   + " for details on how you can specify colormaps for matplotlib")
    #For 3D visualizations, the user can specify an alpha/transparency value in addition to the plot color.
    if plotType == constants.PLOT_TYPE_3D:
        #Check if the user has provided an alpha value
        if plotAlphaString in plotPar.keys():
            #Make sure that the alpha value is a float (or int, if the alpha value is 1)) > 0 and <= 1.
            if type(plotPar[plotAlphaString]) != float and type(plotPar[plotAlphaString]) != int:
                raise RuntimeError("VISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum+1) + "):\n"
                                   + "You did not pass a numerical value for " + plotAlphaString + " "
                                   + "in your visualization parameters structure. Please only "
                                   + "submit floating point values between 0 and 1 for plotAlpha "
                                   + "parameters (i.e., the transparency factor for 3D visualizations).")
            elif plotPar[plotAlphaString] <= 0:
                raise RuntimeError("VISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum) + "):\n"
                                   + "Your value for " + plotAlphaString + " in your visualization parameters "
                                   + "structure is too small (<= 0). The plotAlpha factor for a 3D visualization "
                                   + "must be greater than 0 and less than or equal to 1.0. The default plotAlpha "
                                   + "value in PythonMHD is " + str(constants.DEFAULT_PLOT_ALPHA) + ".")
            elif plotPar[plotAlphaString] > 1:
                raise RuntimeError("VISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum) + "):\n"
                                   + "Your value for " + plotAlphaString + " in your visualization parameters "
                                   + "structure is too large (> 1). The plotAlpha factor for a 3D visualization "
                                   + "must be greater than 0 and less than or equal to 1.0. The default plotAlpha "
                                   + "value in PythonMHD is " + str(constants.DEFAULT_PLOT_ALPHA) + ".")

######PLOT MIN/MAX VALIDATOR######
#Function: checkPlotMinMaxValues
#Purpose: Checks whether the user has requested valid min/max values for their data plot.
#Inputs: plotPar (the user's parameters dictionary for the plot)
#        figNum (the figure number (so we can tell the user which figure is causing an error, if there is one))
#        plotNum (the plot number (so we can tell the user which plot is causing an error, if there is one))
#Outputs: None (if validation is successful)
#         RuntimeError if the min/max values are invalid
def checkPlotMinMaxValues(plotPar,figNum, plotNum):
    #Get some parameter string constants for reading values from plotPar
    minPlotValString = constants.MIN_PLOT_VAL
    maxPlotValString = constants.MAX_PLOT_VAL
    #Make sure minPlotVal and maxPlotVal are numeric values (if the user has specified them)
    if minPlotValString in plotPar.keys():
        if type(plotPar[minPlotValString]) != int and type(plotPar[minPlotValString]) != float:
            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum) + "):\n"
                               + "You passed an invalid value for " + minPlotValString + " in your "
                               + "visualization parameters structure.\nPlease pass a numeric value for "
                               + minPlotValString + " or remove it from your visPar dictionary.")
    if maxPlotValString in plotPar.keys():
        if type(plotPar[maxPlotValString]) != int and type(plotPar[maxPlotValString]) != float:
            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum) + "):\n"
                           + "You passed an invalid value for " + maxPlotValString + " in your "
                               + "visualization parameters structure.\nPlease pass a numeric value for "
                               + maxPlotValString + " or remove it from your visPar dictionary.")
    #If the user has specified both a minPlotVal and a maxPlotVal, we need to make sure that
    #the maxPlotVal is larger than minPlotVal.
    if minPlotValString in plotPar.keys() and maxPlotValString in plotPar.keys():
        minPlotVal = plotPar[minPlotValString]
        maxPlotVal = plotPar[maxPlotValString]
        if maxPlotVal <= minPlotVal:
            raise RuntimeError("\nVISUALIZATION ERROR (Figure "+ str(figNum+1) + ", Plot " + str(plotNum) + "):\n"
                               + "Your maxPlotVal value is less than or equal to your minPlotVal value.\n"
                               + "Please reduce minPlotVal, increase maxPlotVal, or remove either of these values from "
                               + "visPar.")

# ######VISUALIZATION RUNTIME PARAMETERS VALIDATOR######
# # Function: checkVisRuntimePar
# # Purpose: Checks whether the user has requested valid runtime parameters for their visualizations.
# # Inputs: visPar (the user's parameters dictionary for the visualizations)
# # Outputs: None (if validation is successful)
# #         RuntimeError if the runtime parameters are invalid
# def checkVisRuntimePar(visPar):
#     #If the user has passed the "plotPause" parameter for how long PythonMHD should pause after generating new
#     #visualizations, we need to make sure that they have passed an integer or float > 0.
#     if "plotPause" in visPar.keys():
#         if type(visPar["plotPause"]) != int and type(visPar["plotPause"]) != float:
#             raise RuntimeError("You passed an invalid value for plotPause in your visualization parameters structure. "
#                                + "Please only pass numeric values for plotPause (the amount of time in seconds "
#                                + "for which you want to pause the sim after plotting data). You will probably only "
#                                + "need a plotPause value if you are running a 1D simulation. For 2D and 3D simulations, "
#                                + "the longer calculation times should prevent the visualizations from being updated "
#                                + "too quickly.")
#         elif visPar["plotPause"] <= 0:
#             raise RuntimeError("You passed an invalid value for plotPause in your visualization parameters structure. "
#                                + "If you specify a plotPause time, it must be greater than zero. Remove plotPause "
#                                + "from your visualization structure if you don't want their to be an extra pause "
#                                + "between visualization updates. (There will always be a delay between visualizations "
#                                + "while PythonMHD calculates the new simulation state.)")

######FULL VALIDATOR FUNCTION######
# Function: validateVisPar
# Purpose: Checks whether all of the user's settings in their visPar structure are valid in PythonMHD.
#          (This function is the one that gets called in the SimVisualizer constructor when PythonMHD
#           first sets up the user's matplotlib figures.)
# Inputs: simulationGrid (the SimulationGrid object that contains the data for the simulation)
#         visPar (the user's parameters dictionary for the visualizations)
# Outputs: None (if validation is successful)
#          RuntimeError if any of the parameters are invalid
def validateVisPar(simulationGrid, visPar, savingFigs):
    print("Validating Visualization Parameters\n")
    #Get some parameter string constants that will help us read values from visPar
    figuresString = constants.FIGURES
    plotsString = constants.PLOTS
    #Make sure that visPar is a dictionary
    if not isinstance(visPar,dict):
        raise RuntimeError("\nYou passed an invalid data type for the visualization parameters structure (visPar)."
                           + "\nThe visualization parameters must be passed to PythonMHD as a dictionary object "
                           + "\n(see PythonMHD user guide for examples of how to properly structure visPar).")
    #Validate the overall runtime parameters
    #checkVisRuntimePar(visPar) #this version of PythonMHD doesn't have any runtime variables to check
    #Make sure that the user is trying to create at least one matplotlib figure
    if figuresString not in visPar.keys():
        raise RuntimeError("\nIf plotData is set to True in your simPar structure, you must define at least one "
                           "figure in visPar.\nSee the PythonMHD user guide for how to set up your visualizations "
                           "with the visPar structure.")
    elif not isinstance(visPar[figuresString],list):
        #Make sure that the user has passed a list of the figures they want to produce
        #(the list will only have one entry if they only want one plot)
        raise RuntimeError("\nThe " + constants.FIGURES + " parameter in visPar must be a list/array of the figures that you want to "
                           + "create for the simulation.\nSee the PythonMHD user guide for how to define these figures "
                           + "in visPar.")
    else: #if the user has provided a list object for the "figures" parameter
        #Get the number of figures that they want to create
        numFigures = len(visPar[figuresString])
        #Make sure they want to create at least one figure
        if numFigures < 1:
            raise RuntimeError("\nIf " + constants.PLOT_DATA+ " is set to True in your simPar structure, you must request at least one "
                                + "matplotlib figure in your visPar structure.\nSee the PythonMHD user guide for how to "
                                + "set up your visualizations with the visPar structure.")
        #Iterate over all of the figures that the user has requested
        for figNum in range(numFigures):
            #Get the parameters for the current figure
            figPar = visPar[figuresString][figNum]
            #Make sure that they have provided a list of one or more data plots to display on the figure
            if plotsString not in visPar[figuresString][figNum]:
                raise RuntimeError("\nFigure #" + str(figNum+1) + " in your visPar structure does not have a plots list/array. "
                                   + "\nSee the PythonMHD user guide for how to define the data plots for each figure "
                                   + "that you request in visPar.")
            elif not isinstance(figPar[plotsString], list):
                raise RuntimeError("\nThe plots parameter in visPar must be a list/array of the data plots that you want "
                                   + "to have in a matplotlib figure.\nSee the PythonMHD user guide for how to create "
                                   + "the plots list/array.")
            else:
                #Check if the number of plots is >= 1 and < the max number of plots that can be displayed on a
                #PythonMHD matplotlib figure
                numPlots = len(figPar[plotsString])
                if numPlots < 1:
                    raise RuntimeError("\nFigure #" + str(figNum+1) + " in your visPar structure does not have any "
                                       + "requested data plots.\nYou must have at least one data plot on each "
                                       + "matplotlib figure that you want to create.\nSee the PythonMHD user guide "
                                       + "for to set up the plots list/array for each figure that you define in visPar.")
                elif numPlots > constants.MAX_PLOTS:
                    raise RuntimeError("\nFigure #" + str(figNum+1) + " has too many requested data plots in visPar.\n"
                                       + "The maximum number of plots that can be displayed on one figure is "
                                       + str(constants.MAX_PLOTS) + ".\n\nIf you need more than " + str(constants.MAX_PLOTS) + " "
                                       + "data plots, please define multiple figures in visPar\n(see PythonMHD user guide "
                                       + "for additional details).\n\nAlternatively, if you want to override the default "
                                       + "max number of plots per figure,\nchange the MAX_PLOTS constant in the file "
                                       + "PythonMHD_Constants.py. However,\nplease be aware that the default value was "
                                       + "chosen to ensure that all data plots\non a figure are large enough to easily "
                                       + "interpret.")
                else: #if the number of plots is valid
                    #If the user has specified an overall title for the figure, make sure it is a string with a
                    #reasonable length.
                    if constants.FIGURE_TITLE in figPar.keys():
                        if not isinstance(figPar[constants.FIGURE_TITLE], str):
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum+1) + "):\nYou passed an "
                                               + "invalid value for " + constants.FIGURE_TITLE + " in visPar. "
                                               + "\n\nIf you want to specify the title for the figure, you must pass a "
                                               + "string\n(with fewer than or exactly " + str(constants.MAX_FIGURE_TITLE_CHARACTERS)
                                               + " characters) for the " + constants.FIGURE_TITLE + " parameter.")
                        elif len(figPar[constants.FIGURE_TITLE]) > constants.MAX_FIGURE_TITLE_CHARACTERS:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "):\nYou passed a "
                                               + "string that is too long for " + constants.FIGURE_TITLE
                                               + " in visPar.\nThe max number of characters for a figure title is "
                                               + str(constants.MAX_FIGURE_TITLE_CHARACTERS) + ".\n\nYou can override this "
                                               + "limit by changing the value of the MAX_FIGURE_TITLE_CHARACTERS constant "
                                               + "in PythonMHD_Constants.py.")
                    #Figure out if we are saving the images generated in this figure
                    #(the user might want to override the overall saveFigs flag for all figures)
                    savingImages = savingFigs
                    if constants.SAVE_FIGS in figPar.keys():
                        #Make sure that their saveFigs flag is a boolean
                        if type(figPar[constants.SAVE_FIGS]) != bool:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum+1) + "):\nYou passed an "
                                               + "invalid value for " + constants.SAVE_FIGS + " in visPar.\n"
                                               + "The only accepted values for " + constants.SAVE_FIGS + " are True or False.\n\nSet " + constants.SAVE_FIGS + " to True "
                                               + "for this figure if you want to save all of its visualization outputs. "
                                               + "\nIf you remove " + constants.SAVE_FIGS + " from visPar, PythonMHD will use your overall " + constants.SAVE_FIGS + " "
                                               + "parameter in simPar\n(or the default value of " + str(constants.DEFAULT_SAVE_FIGS)
                                               + ") if you do not have a " + constants.SAVE_FIGS + " parameter in simPar).")
                        else:
                            savingImages = figPar[constants.SAVE_FIGS] and savingFigs
                    if savingImages:
                        #Check if the user wants to make a movie with their saved visualizations
                        if constants.MAKE_MOVIE in figPar.keys():
                            #Make sure that their makeMovie flag is a boolean
                            if type(figPar[constants.MAKE_MOVIE]) != bool:
                                raise RuntimeError("\nYou passed an invalid value for " + constants.MAKE_MOVIE + " in your visualization parameters structure. "
                                                   + "\n\nThe only accepted values for " + constants.MAKE_MOVIE + " are True or False.\nSet " + constants.MAKE_MOVIE + " to True if you "
                                                   + "save all of your visualization outputs.\n\nThe default value of " + constants.MAKE_MOVIE + " is False, which "
                                                   + "which means that PythonMHD will assume\nthat you do not need movies of your "
                                                   + "visualizations if you remove the " + constants.MAKE_MOVIE + " parameter from visPar and simPar.")
                        #Check if the figure has an output image format parameter
                        if constants.SAVE_FIGS_FORMAT in figPar.keys():
                            #For now, just make sure that the output format parameter
                            #(if one has been supplied) is a string (whether the output
                            #format itself is valid will depend on the matplotlib backend).
                            #The rest of the file format verification will take place in
                            #the SimVisualizer class.
                            if not isinstance(figPar[constants.SAVE_FIGS_FORMAT],str):
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "):\nYou passed an "
                                                   + "invalid data type for " + constants.SAVE_FIGS_FORMAT + " in your simulation "
                                                   + "parameters structure.\nYou must pass a string value for the image format parameter "
                                                   + "(e.g., \"png\", \"pdf\", etc.).\nThe supported file types will depend on your "
                                                   + "matplotlib backend (see PythonMHD user guide for more details).")
                            elif constants.MAKE_MOVIE in figPar.keys():
                                if figPar[constants.MAKE_MOVIE] and figPar[constants.SAVE_FIGS_FORMAT] != constants.IMAGE_FORMAT_PNG:
                                    raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "):\nYou passed an "
                                                       + "invalid value for " + constants.SAVE_FIGS_FORMAT + " in your "
                                                       + "visualization parameters structure.\nIf you want to make movies of "
                                                       + "your simulation data, you need to save the visualization outputs as .png files.")
                        #Check if the figure has an output image resolution (DPI/dots-per-inch)
                        if constants.SAVE_FIGS_DPI in figPar.keys():
                            #Make sure that the output image resolution (if one has been specified), is an
                            #integer greater than zero. PythonMHD will warn the user, but not raise an error,
                            #if the number of dots-per-inch is outside the recommend range.
                            if type(figPar[constants.SAVE_FIGS_DPI]) != int:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "): You passed an "
                                                   + "invalid data type for " + constants.SAVE_FIGS_DPI
                                                   + " in your visualization parameters.\nThe number of dots per inch (dpi) "
                                                   + "in your output figures must be an integer value (e.g., 300, 450, etc.). "
                                                   + "\nIf you remove the dpi parameter from visPar and simPar, PythonMHD will use "
                                                   + "a default resolution of " + str(constants.DEFAULT_SAVE_FIGS_DPI) + ".")
                            elif figPar[constants.SAVE_FIGS_DPI] <= 0:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "): You passed an "
                                                   + "invalid value for " + constants.SAVE_FIGS_DPI
                                                   + " in your visualization parameters. The number of dots per inch (dpi) "
                                                   + "in your output figures must be an integer value greater than zero. "
                                                   + "If you remove the dpi parameter from visPar and simPar, PythonMHD will use "
                                                   + "a default resolution of " + str(constants.DEFAULT_SAVE_FIGS_DPI) + "."
                                                   + "In most cases, you will want to use at least this resolution to "
                                                   + "produce high-quality images.")
                            elif figPar[constants.SAVE_FIGS_DPI] < constants.MIN_RECOMMENDED_SAVE_FIGS_DPI:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "\nYou passed "
                                                   + "a value for " + constants.SAVE_FIGS_DPI
                                                   + " in your visualization parameters that is below the minimum "
                                                   + "\nrecommended resolution for output images ("
                                                   + str(constants.MIN_RECOMMENDED_SAVE_FIGS_DPI) + " dots per inch).\n\n"
                                                   + "If you remove the dpi parameter from visPar and simPar, PythonMHD will use "
                                                   + "a default resolution of " + str(constants.DEFAULT_SAVE_FIGS_DPI) + "."
                                                   + "\nIn most cases, you will want to use at least this resolution to "
                                                   + "produce high-quality images.")
                            elif figPar[constants.SAVE_FIGS_DPI] > constants.MAX_RECOMMENDED_SAVE_FIGS_DPI:
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "\nYou passed "
                                                   + "a value for " + constants.SAVE_FIGS_DPI
                                                   + " in your visualization parameters that is above the maximum "
                                                   + "\nrecommended resolution for output images ("
                                                   + str(constants.MAX_RECOMMENDED_SAVE_FIGS_DPI) + " dots per inch)."
                                                   + "\n\nIf you remove the dpi parameter from visPar and simPar, PythonMHD will use "
                                                   + "a default resolution of " + str(constants.DEFAULT_SAVE_FIGS_DPI) + "."
                                                   + "\nIn most cases, a resolution closer to this value will be "
                                                   + "sufficient to produce high-quality images.\nIf you make the dpi too large, "
                                                   + "the output images will take up an excessive amount of space on your computer.")
                    #Validate the user's values for numSubPlotRows and numSubPlotCols
                    #(if they specified these parameters in visPar)
                    #Both numSubPlotRows and numSubPlotCols must be >= 1
                    if constants.FIGURE_NUM_ROWS in figPar.keys():
                        if type(figPar[constants.FIGURE_NUM_ROWS]) != int:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "):\nYou passed an "
                                               + "invalid data type for " + constants.FIGURE_NUM_ROWS + " in your "
                                               + "visualization parameters structure.\nThe number of rows in your matplotlib "
                                               + "figure must be an integer value greater than 0.\n")
                        if figPar[constants.FIGURE_NUM_ROWS] < 1:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "):\nYou passed a "
                                + "value for " + constants.FIGURE_NUM_ROWS + " that is too small.\nThe minimum "
                                + "number of rows is 1.")
                    if constants.FIGURE_NUM_COLS in figPar.keys():
                        if type(figPar[constants.FIGURE_NUM_COLS]) != int:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "):\nYou passed an "
                                               + "invalid data type for " + constants.FIGURE_NUM_COLS + " in your "
                                               + "visualization parameters structure.\nThe number of columns in your matplotlib "
                                               + "figure must be an integer value greater than 0.\n")
                        if figPar[constants.FIGURE_NUM_COLS] < 1:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "):\nYou passed a "
                                + "value for " + constants.FIGURE_NUM_COLS + " that is too small.\nThe minimum "
                                + "number of columns is 1.")
                    #Make sure that numSubPlotRows*numSubPlotCols (the total number of spaces for plots) is consistent
                    #with the number of plots that the user wants to display on the figure
                    if constants.FIGURE_NUM_ROWS in figPar.keys() and constants.FIGURE_NUM_COLS in figPar.keys():
                        numSubPlotRows = figPar[constants.FIGURE_NUM_ROWS]
                        numSubPlotCols = figPar[constants.FIGURE_NUM_COLS]
                        #Ensure that there are enough spaces to accommodate all of the plots in the figure
                        if numSubPlotRows*numSubPlotCols < numPlots:
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "):\nYou passed "
                                + "invalid values for " + constants.FIGURE_NUM_ROWS + " and " + constants.FIGURE_NUM_COLS
                                + " in your visualization parameters structure,\nbecause the product of the number of rows "
                                + "and the number of columns is less than the number of plots that \nyou requested "
                                +  "(numRows*numCols = " + str(numSubPlotRows*numSubPlotCols) + " < numPlots = "
                                + str(numPlots) + ").")
                        elif numPlots <= numSubPlotCols*(numSubPlotRows-1):
                            #We also don't want there to be too many spaces for the number of plots.
                            #Here we make sure that there is never a completely empty row in the matplotlib figure.
                            raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum + 1) + "):\nYou passed "
                                + "invalid values for " + constants.FIGURE_NUM_ROWS + " and " + constants.FIGURE_NUM_COLS
                                + " in your visualization parameters structure,\nbecause the product of the number of rows "
                                + "and the number of columns is too large for the number of plots that \nyou requested "
                                +  "(numRows*numCols = " + str(numSubPlotRows*numSubPlotCols) + " for "+str(numPlots)
                                + " plots will leave at least one completely empty row in your figure).")

                    #Iterate over the plots in the figure
                    for plotNum in range(numPlots):
                        #Get the parameters for the current plot
                        plotPar = figPar[plotsString][plotNum]
                        #Make sure the user has requested a valid hydrodynamic or MHD variable
                        if constants.PLOT_VAR in plotPar.keys():
                            if not isinstance(plotPar[constants.PLOT_VAR],str):
                                raise RuntimeError("\nVISUALIZATION ERROR (Figure #" + str(figNum+1) + ", Plot #"
                                                   + str(plotNum+1) + "):\nYou passed an invalid "
                                                  + "value for " + constants.PLOT_VAR + " in visPar. You must pass "
                                                  + "a string value\nfor the gas variable that should be visualized "
                                                  + "(e.g., " + constants.DENSITY + ", " + constants.PRESSURE
                                                  + ", " + constants.ENERGY + ", etc.).")
                            if simulationGrid.isMHD:
                                checkMHDPlotVar(plotPar[constants.PLOT_VAR], figNum, plotNum)
                            else:
                                checkHydroPlotVar(plotPar[constants.PLOT_VAR], figNum, plotNum)
                        #Validate the plot type
                        checkPlotType(plotPar, simulationGrid, figNum, plotNum)
                        #Validate the plot color
                        checkPlotColor(plotPar, simulationGrid.nDim, figNum, plotNum)
                        #Validate the min/max values for the plot
                        checkPlotMinMaxValues(plotPar,  figNum, plotNum)






