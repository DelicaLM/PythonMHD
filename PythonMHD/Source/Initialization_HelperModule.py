#Initialization_HelperModule.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides supporting functions for PythonMHD initialization files.

######IMPORT STATEMENTS######

#Import NumPy for matrix operations
import numpy as np

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants

#######SIMULATION GRID COORDINATE FUNCTIONS#######
#The functions below allow the user to easily obtain the coordinates that correspond to their grid parameters
#in a PythonMHD initialization file.
#Function: createCoordinates1D_Cartesian
#Purpose: Creates Cartesian coordinate matrix for a 1D simulation based on the
#         parameters in gridPar.
#Input Parameters: gridPar (the grid parameters dictionary that should
#                           be used for constructing the coordinate
#                           matrices)
#Outputs: xCoords (a matrix with the x-coordinates of every cell
#                  in the simulation grid)
def createCoordinates1D_Cartesian(gridPar):
    #Make sure that gridPar has a numXCells value
    if not (constants.NUM_X_CELLS in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.NUM_X_CELLS + " in your grid parameters dictionary.")
    #Make sure that numXCells is an int
    if not (isinstance(gridPar[constants.NUM_X_CELLS], int)):
        raise RuntimeError("\nThe " + constants.NUM_X_CELLS + " value in your grid parameters dictionary is not an integer.")
    #Make sure that there are at least two cells in the x-direction
    if gridPar[constants.NUM_X_CELLS] < 1:
        raise RuntimeError("\nThe " + constants.NUM_X_CELLS + " value in your grid parameters dictionary is too small." +
                           "\nYou need at least two cells in the x-direction for a 1D simulation.")
    #Make sure that minX and maxX are defined and valid
    if not(constants.MIN_X in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MIN_X + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MIN_X]) != int and type(gridPar[constants.MIN_X]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MIN_X + " in your grid parameters dictionary."
                           +"\nThe minimum x-coordinate must be a numeric value (i.e., an integer or a float).")
    elif not(constants.MAX_X in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MAX_X + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MAX_X]) != int and type(gridPar[constants.MAX_X]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MAX_X + " in your grid parameters dictionary."
                           +"\nThe maximum x-coordinate must be a numeric value (i.e., an integer or a float).")
    elif gridPar[constants.MIN_X] >= gridPar[constants.MAX_X]:
        raise RuntimeError("\nInvalid values for " + constants.MIN_X + " and " + constants.MAX_X + ".\nThe minimum "
                           + "x-coordinate (" + constants.MIN_X + ") must be smaller than the maximum x-coordinate "
                           + "(" + constants.MAX_X + ").")
    #Get the width of each cell
    dx = (gridPar[constants.MAX_X] - gridPar[constants.MIN_X])/gridPar[constants.NUM_X_CELLS]
    #Create the x-coordinates vector
    xCoords = np.r_[gridPar[constants.MIN_X] + dx/2.0:gridPar[constants.MAX_X] + dx/2.0:dx].reshape(-1)
    return xCoords

#Function: createCoordinates2D_Cartesian
#Purpose: Creates Cartesian coordinate matrix for a 2D simulation based on the
#         parameters in gridPar.
#Input Parameters: gridPar (the grid parameters dictionary that should
#                           be used for constructing the coordinate
#                           matrices)
#Outputs: xCoords (a matrix with the x-coordinates of every cell
#                  in the simulation grid)
#         yCoords (a matrix with the y-coordinates of every cell
#                  in the simulation grid)
def createCoordinates2D_Cartesian(gridPar):
    #Make sure that gridPar has a numXCells value
    if not (constants.NUM_X_CELLS in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.NUM_X_CELLS + " in your grid parameters dictionary.")
    #Make sure that numXCells is an int
    if not (isinstance(gridPar[constants.NUM_X_CELLS], int)):
        raise RuntimeError("\nThe " + constants.NUM_X_CELLS + " value in your grid parameters dictionary is not an integer.")
    #Make sure that there are at least two cells in the x-direction
    if gridPar[constants.NUM_X_CELLS] < 1:
        raise RuntimeError("\nThe " + constants.NUM_X_CELLS + " value in your grid parameters dictionary is too small." +
                           "\nYou need at least two cells in the x-direction for a 2D simulation.")
    #Make sure that gridPar has a numYCells value
    if not (constants.NUM_Y_CELLS in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.NUM_Y_CELLS + " in your grid parameters dictionary.")
    #Make sure that numYCells is an int
    if not (isinstance(gridPar[constants.NUM_Y_CELLS], int)):
        raise RuntimeError("\nThe " + constants.NUM_Y_CELLS + " value in your grid parameters dictionary is not an integer.")
    #Make sure that there are at least two cells in the y-direction
    if gridPar[constants.NUM_Y_CELLS] < 1:
        raise RuntimeError("\nThe " + constants.NUM_Y_CELLS + " value in your grid parameters dictionary is too small." +
                           "\nYou need at least two cells in the y-direction for a 2D simulation.")
    #Make sure that minX and maxX are defined and valid
    if not(constants.MIN_X in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MIN_X + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MIN_X]) != int and type(gridPar[constants.MIN_X]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MIN_X + " in your grid parameters dictionary."
                           +"\nThe minimum x-coordinate must be a numeric value (i.e., an integer or a float).")
    elif not(constants.MAX_X in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MAX_X + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MAX_X]) != int and type(gridPar[constants.MAX_X]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MAX_X + " in your grid parameters dictionary."
                           +"\nThe maximum x-coordinate must be a numeric value (i.e., an integer or a float).")
    elif gridPar[constants.MIN_X] >= gridPar[constants.MAX_X]:
        raise RuntimeError("\nInvalid values for " + constants.MIN_X + " and " + constants.MAX_X + ".\nThe minimum "
                           + "x-coordinate (" + constants.MIN_X + ") must be smaller than the maximum x-coordinate "
                           + "(" + constants.MAX_X + ").")
    #Make sure that minY and maxY are defined and valid
    if not(constants.MIN_Y in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MIN_Y + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MIN_Y]) != int and type(gridPar[constants.MIN_Y]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MIN_Y + " in your grid parameters dictionary."
                           +"\nThe minimum y-coordinate must be a numeric value (i.e., an integer or a float).")
    elif not(constants.MAX_Y in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MAX_Y + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MAX_Y]) != int and type(gridPar[constants.MAX_Y]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MAX_Y + " in your grid parameters dictionary."
                           +"\nThe maximum y-coordinate must be a numeric value (i.e., an integer or a float).")
    elif gridPar[constants.MIN_Y] >= gridPar[constants.MAX_Y]:
        raise RuntimeError("\nInvalid values for " + constants.MIN_Y + " and " + constants.MAX_Y + ".\nThe minimum "
                           + "y-coordinate (" + constants.MIN_Y + ") must be smaller than the maximum y-coordinate "
                           + "(" + constants.MAX_Y + ").")
    #Get the width of each cell
    dx = (gridPar[constants.MAX_X] - gridPar[constants.MIN_X])/gridPar[constants.NUM_X_CELLS]
    #Get the height of each cell
    dy = (gridPar[constants.MAX_Y] - gridPar[constants.MIN_Y])/gridPar[constants.NUM_Y_CELLS]
    #Create the x-coordinates matrix
    xCoords = np.r_[gridPar[constants.MIN_X] + dx/2.0:gridPar[constants.MAX_X] + dx/2.0:dx].reshape(1,-1)
    xCoords = np.repeat(xCoords.reshape(1,-1), gridPar[constants.NUM_Y_CELLS], axis=0)
    #Create the y-coordinates matrix
    yCoords = np.r_[gridPar[constants.MIN_Y] + dy/2.0:gridPar[constants.MAX_Y] + dy/2.0:dy].reshape(-1,1)
    yCoords = np.repeat(yCoords, gridPar[constants.NUM_X_CELLS], axis=1)
    return (xCoords, yCoords)

#Function: createCoordinates3D_Cartesian
#Purpose: Creates Cartesian coordinate matrix for a 3D simulation based on the
#         parameters in gridPar.
#Input Parameters: gridPar (the grid parameters dictionary that should
#                           be used for constructing the coordinate
#                           matrices)
#Outputs: xCoords (a matrix with the x-coordinates of every cell
#                  in the simulation grid)
#         yCoords (a matrix with the y-coordinates of every cell
#                  in the simulation grid)
#         zCoords (a matrix with the z-coordinates of every cell
#                  in the simulation grid)
def createCoordinates3D_Cartesian(gridPar):
    #Make sure that gridPar has a numXCells value
    if not (constants.NUM_X_CELLS in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.NUM_X_CELLS + " in your grid parameters dictionary.")
    #Make sure that numXCells is an int
    if not (isinstance(gridPar[constants.NUM_X_CELLS], int)):
        raise RuntimeError("\nThe " + constants.NUM_X_CELLS + " value in your grid parameters dictionary is not an integer.")
    #Make sure that there are at least two cells in the x-direction
    if gridPar[constants.NUM_X_CELLS] < 1:
        raise RuntimeError("\nThe " + constants.NUM_X_CELLS + " value in your grid parameters dictionary is too small." +
                           "\nYou need at least two cells in the x-direction for a 3D simulation.")
    #Make sure that gridPar has a numYCells value
    if not (constants.NUM_Y_CELLS in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.NUM_Y_CELLS + " in your grid parameters dictionary.")
    #Make sure that numYCells is an int
    if not (isinstance(gridPar[constants.NUM_Y_CELLS], int)):
        raise RuntimeError("\nThe " + constants.NUM_Y_CELLS + " value in your grid parameters dictionary is not an integer.")
    #Make sure that there are at least two cells in the y-direction
    if gridPar[constants.NUM_Y_CELLS] < 1:
        raise RuntimeError("\nThe " + constants.NUM_Y_CELLS + " value in your grid parameters dictionary is too small." +
                           "\nYou need at least two cells in the y-direction for a 3D simulation.")
    #Make sure that gridPar has a numZCells value
    if not (constants.NUM_Z_CELLS in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.NUM_Z_CELLS + " in your grid parameters dictionary.")
    #Make sure that numZCells is an int
    if not (isinstance(gridPar[constants.NUM_Z_CELLS], int)):
        raise RuntimeError("\nThe " + constants.NUM_Z_CELLS + " value in your grid parameters dictionary is not an integer.")
    #Make sure that there are at least two cells in the z-direction
    if gridPar[constants.NUM_Z_CELLS] < 1:
        raise RuntimeError("\nThe " + constants.NUM_Z_CELLS + " value in your grid parameters dictionary is too small." +
                           "\nYou need at least two cells in the z-direction for a 3D simulation.")
    #Make sure that minX and maxX are defined and valid
    if not(constants.MIN_X in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MIN_X + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MIN_X]) != int and type(gridPar[constants.MIN_X]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MIN_X + " in your grid parameters dictionary."
                           +"\nThe minimum x-coordinate must be a numeric value (i.e., an integer or a float).")
    elif not(constants.MAX_X in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MAX_X + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MAX_X]) != int and type(gridPar[constants.MAX_X]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MAX_X + " in your grid parameters dictionary."
                           +"\nThe maximum x-coordinate must be a numeric value (i.e., an integer or a float).")
    elif gridPar[constants.MIN_X] >= gridPar[constants.MAX_X]:
        raise RuntimeError("\nInvalid values for " + constants.MIN_X + " and " + constants.MAX_X + ".\nThe minimum "
                           + "x-coordinate (" + constants.MIN_X + ") must be smaller than the maximum x-coordinate "
                           + "(" + constants.MAX_X + ").")
    #Make sure that minY and maxY are defined and valid
    if not(constants.MIN_Y in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MIN_Y + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MIN_Y]) != int and type(gridPar[constants.MIN_Y]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MIN_Y + " in your grid parameters dictionary."
                           +"\nThe minimum y-coordinate must be a numeric value (i.e., an integer or a float).")
    elif not(constants.MAX_Y in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MAX_Y + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MAX_Y]) != int and type(gridPar[constants.MAX_Y]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MAX_Y + " in your grid parameters dictionary."
                           +"\nThe maximum y-coordinate must be a numeric value (i.e., an integer or a float).")
    elif gridPar[constants.MIN_Y] >= gridPar[constants.MAX_Y]:
        raise RuntimeError("\nInvalid values for " + constants.MIN_Y + " and " + constants.MAX_Y + ".\nThe minimum "
                           + "y-coordinate (" + constants.MIN_Y + ") must be smaller than the maximum y-coordinate "
                           + "(" + constants.MAX_Y + ").")
    #Make sure that minZ and maxZ are defined and valid
    if not(constants.MIN_Z in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MIN_Z + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MIN_Z]) != int and type(gridPar[constants.MIN_Z]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MIN_Z + " in your grid parameters dictionary."
                           +"\nThe minimum z-coordinate must be a numeric value (i.e., an integer or a float).")
    elif not(constants.MAX_Z in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.MAX_Z + " in your grid parameters dictionary.")
    elif type(gridPar[constants.MAX_Z]) != int and type(gridPar[constants.MAX_Z]) != float:
        raise RuntimeError("\nYou passed an invalid data type for " + constants.MAX_Z + " in your grid parameters dictionary."
                           +"\nThe maximum z-coordinate must be a numeric value (i.e., an integer or a float).")
    elif gridPar[constants.MIN_Z] >= gridPar[constants.MAX_Z]:
        raise RuntimeError("\nInvalid values for " + constants.MIN_Z + " and " + constants.MAX_Z + ".\nThe minimum "
                           + "z-coordinate (" + constants.MIN_Z + ") must be smaller than the maximum z-coordinate "
                           + "(" + constants.MAX_Z + ").")
    #Get the width of each cell
    dx = (gridPar[constants.MAX_X] - gridPar[constants.MIN_X])/gridPar[constants.NUM_X_CELLS]
    #Get the height of each cell
    dy = (gridPar[constants.MAX_Y] - gridPar[constants.MIN_Y])/gridPar[constants.NUM_Y_CELLS]
    # Get the depth of each cell
    dz = (gridPar[constants.MAX_Z] - gridPar[constants.MIN_Z])/gridPar[constants.NUM_Z_CELLS]
    #Create the x-coordinates matrix
    xCoords = np.r_[gridPar[constants.MIN_X] + dx/2.0:gridPar[constants.MAX_X] + dx/2.0:dx].reshape(1,-1,1)
    xCoords = np.repeat(xCoords, gridPar[constants.NUM_Y_CELLS], axis=0)
    xCoords = np.repeat(xCoords, gridPar[constants.NUM_Z_CELLS], axis=2)
    #Create the y-coordinates matrix
    yCoords = np.r_[gridPar[constants.MIN_Y] + dy/2.0:gridPar[constants.MAX_Y] + dy/2.0:dy].reshape(-1,1,1)
    yCoords = np.repeat(yCoords, gridPar[constants.NUM_X_CELLS], axis=1)
    yCoords = np.repeat(yCoords, gridPar[constants.NUM_Z_CELLS], axis=2)
    #Create the z-coordinates matrix
    zCoords = np.r_[gridPar[constants.MIN_Z] + dz/2.0:gridPar[constants.MAX_Z] + dz/2.0:dz].reshape(1,1,-1)
    zCoords = np.repeat(zCoords, gridPar[constants.NUM_Y_CELLS], axis=0)
    zCoords = np.repeat(zCoords, gridPar[constants.NUM_X_CELLS], axis=1)
    return (xCoords, yCoords, zCoords)

