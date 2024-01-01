#SimulationGrid_HelperModule.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides supporting functions (e.g., validating user parameters, creating coordinate matrices)
#         for the SimulationGrid class in SimulationGrid.py.

######IMPORT STATEMENTS#######

#Import Numpy for matrix operations
import numpy as np

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants


######GRID PARAMETERS VALIDATION FUNCTION
#Function: validateGridPar
#Purpose: Checks whether the user has submitted valid inputs for gridPar (when compared
#         against the primitive variable matrix). If there are any problems (e.g.,
#         only 1 or more than 4 dimensions in primVars, min coordinate is larger than
#         max coordinate, number of cells in primVars doesn't match the number of x, y,
#         and z cells in gridPar), a RuntimeError will be raised.
#Input Parameters: primVars (the primitive variables matrix)
#                  gridPar (the grid parameters dictionary to check)
#Outputs: void (this function will raise a RuntimeError if gridPar
#               fails the validation test)
def validateGridPar(primVars, gridPar):
    #Check the number of spatial dimensions in primVars
    #(ignoring any singleton dimensions)
    numDim = len(np.squeeze(primVars).shape) - 1

    #Raise a RuntimeError if there are too few or too many spatial dimensions
    if numDim < 1:
        raise RuntimeError("\nZero spatial dimensions in primitive variable matrix. Simulation must be 1D, 2D, or 3D.")
    if numDim > 3:
        raise RuntimeError("\nToo many spatial dimensions in primitive variable matrix. Simulation must be 1D, 2D, or 3D.")

    #Note:
    #   For 1D Simulations:
    #       1st spatial dimension: x-direction (horizontal)
    #
    #   For 2D Simulations:
    #       1st spatial dimension: y-direction (vertical)
    #       2nd spatial dimension: x-direction (horizontal)
    #
    #   For 3D Simulations:
    #       1st spatial dimension: y-direction (vertical)
    #       2nd spatial dimension: x-direction (horizontal)
    #       3rd spatial dimension: z-direction (depth)
    #

    #Check if we have valid values for numXCells, numYCells, and numZCells in gridPar
    #Make sure that gridPar has a numXCells value (because every simulation is at least 1D
    if not(constants.NUM_X_CELLS in gridPar.keys()):
        raise RuntimeError("\nYou did not specify " + constants.NUM_X_CELLS + " in your grid parameters dictionary.")
    #Make sure that numXCells is an int
    if not(isinstance(gridPar[constants.NUM_X_CELLS],int)):
        raise RuntimeError("\nThe " + constants.NUM_X_CELLS + " value in your grid parameters dictionary is not an integer.")
    #Make sure that we have a numYCells value if the simulation is 2D or 3D
    if numDim >= 2:
        if not (constants.NUM_Y_CELLS in gridPar.keys()):
            raise RuntimeError("\nYou did not specify " + constants.NUM_Y_CELLS + " in your grid parameters dictionary.")
        #Make sure that numYCells is an int
        if not (isinstance(gridPar[constants.NUM_Y_CELLS], int)):
            raise RuntimeError("\nThe " + constants.NUM_Y_CELLS + " value in your grid parameters dictionary is not an integer.")
    #Make sure that we have a numZCells value if the simulation is 3D
    if numDim == 3:
        if not (constants.NUM_Z_CELLS in gridPar.keys()):
            raise RuntimeError("\nYou did not specify " + constants.NUM_Z_CELLS + " in your grid parameters dictionary.")
        #Make sure that numZCells is an int
        if not (isinstance(gridPar[constants.NUM_Z_CELLS], int)):
            raise RuntimeError("\nThe " + constants.NUM_Z_CELLS + " value in your grid parameters dictionary is not an integer.")
    #Now we will check if the user has enough cells in the x-, y-, and z-directions
    if numDim == 1: #if the sim is 1D
        #Make sure that there are at least two cells in the x-direction
        if gridPar[constants.NUM_X_CELLS] < 2:
            raise RuntimeError("\nThe " + constants.NUM_X_CELLS + " value in your grid parameters dictionary is too small." +
                               "\nYou need at least two cells in the x-direction for a 1D simulation.")
    if numDim == 2: #if the sim is 2D
        #Make sure that there are at least two cells in the x-direction
        if gridPar[constants.NUM_X_CELLS] < 2:
            raise RuntimeError("\nThe " + constants.NUM_X_CELLS + " value in your grid parameters dictionary is too small." +
                               "\nYou need at least two cells in the x-direction for a 2D simulation.")

        #Make sure that there are at least two cells in the y-direction
        if gridPar[constants.NUM_Y_CELLS] < 2:
            raise RuntimeError("\nThe " + constants.NUM_Y_CELLS + " value in your grid parameters dictionary is too small." +
                               "\nYou need at least two cells in the y-direction for a 2D simulation.")
    if numDim == 3: #if the sim is 3D
        #Make sure that there are at least two cells in the x-direction
        if gridPar[constants.NUM_X_CELLS] < 2:
            raise RuntimeError("\nThe " + constants.NUM_X_CELLS + " value in your grid parameters dictionary is too small." +
                               "\nYou need at least two cells in the y-direction for a 3D simulation.")
        if gridPar[constants.NUM_Y_CELLS] < 2:
            raise RuntimeError("\nThe " + constants.NUM_Y_CELLS + " value in your grid parameters dictionary is too small." +
                               "\nYou need at least two cells in the y-direction for a 3D simulation.")
        if gridPar[constants.NUM_Z_CELLS] < 2:
            raise RuntimeError("\nThe " + constants.NUM_Z_CELLS + " value in your grid parameters dictionary is too small." +
                               "\nYou need at least two cells in the z-direction for a 3D simulation.")

    #For any simulation (1D, 2D, or 3D), we need to check if the
    #number of cells in the x-direction matches the numXCells parameter
    #in gridPar. These RuntimeErrors will let the user know whether
    #they have too many or too few cells in the x-direction.
    tooManyXCells = False
    tooFewXCells = False
    if numDim == 1:
        tooManyXCells = primVars.shape[1] > gridPar[constants.NUM_X_CELLS]
        tooFewXCells = primVars.shape[1] < gridPar[constants.NUM_X_CELLS]
    else:
        tooManyXCells = primVars.shape[2] > gridPar[constants.NUM_X_CELLS]
        tooFewXCells = primVars.shape[2] < gridPar[constants.NUM_X_CELLS]
    if tooManyXCells:
        raise RuntimeError("\nPrimitive variables matrix has too many cells in the x-direction." +
                           "\nPlease adjust the primitive variables matrix or change the value" +
                           " of " + constants.NUM_X_CELLS + "\nin your grid parameters dictionary.")
    elif tooFewXCells:
        raise RuntimeError("\nPrimitive variables matrix does not have enough cells in the x-direction." +
                           "\nPlease adjust the primitive variables matrix or change the value" +
                           " of " + constants.NUM_X_CELLS + "\nin your grid parameters dictionary.")
    #If the simulation is 2D or 3D, we need to make sure that the primitive variables matrix
    #has the right number of cells in the y-direction.
    if numDim >= 2:
        if primVars.shape[1] > gridPar[constants.NUM_Y_CELLS]:
            raise RuntimeError("\nPrimitive variables matrix has too many cells in the y-direction." +
                               "\nPlease adjust the primitive variables matrix or change the value" +
                               " of " + constants.NUM_Y_CELLS + "\nin your grid parameters dictionary.")
        elif primVars.shape[1] < gridPar[constants.NUM_Y_CELLS]:
            raise RuntimeError("\nPrimitive variables matrix does not have enough cells in the y-direction." +
                               "\nPlease adjust the primitive variables matrix or change the value" +
                               " of " + constants.NUM_Y_CELLS + "\nin your grid parameters dictionary.")
    #If the simulation is 3D, we need to make sure that the primitive variables matrix
    #has the right number of cells in the z-direction.
    if numDim == 3:
        if primVars.shape[3] > gridPar[constants.NUM_Z_CELLS]:
            raise RuntimeError("\nPrimitive variables matrix has too many cells in the z-direction." +
                               "\nPlease adjust the primitive variables matrix or change the value" +
                               " of " + constants.NUM_Z_CELLS + "\nin your grid parameters dictionary.")
        elif primVars.shape[3] < gridPar[constants.NUM_Z_CELLS]:
            raise RuntimeError("\nPrimitive variables matrix does not have enough cells in the z-direction." +
                               "\nPlease adjust the primitive variables matrix or change the value" +
                               " of " + constants.NUM_Z_CELLS + "\nin your grid parameters dictionary.")

    #Check if the min and max coordinate values in gridPar make sense
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
        raise RuntimeError("\nInvalid values for " + constants.MIN_X + " and " + constants.MAX_X + ".\nThe minimum x-coordinate ("
                           + constants.MIN_X + ") must be smaller than the maximum x-coordinate (" + constants.MAX_X + ").")
    if numDim >= 2:
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
            raise RuntimeError("\nInvalid values for " + constants.MIN_Y + " and " + constants.MAX_Y + ".\nThe minimum y-coordinate ("
                               + constants.MIN_Y + ") must be smaller than the maximum y-coordinate (" + constants.MAX_Y + ").")
    if numDim == 3:
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
            raise RuntimeError("\nInvalid values for " + constants.MIN_Z + " and " + constants.MAX_Z + ".\nThe minimum z-coordinate ("
                               + constants.MIN_Z + ") must be smaller than the maximum z-coordinate (" + constants.MAX_Z + ").")
    return

#Function: formatPrimVars
#Purpose: Puts the primitive variable matrix in the format that will be expected
#         by the functions in the Simulation class. For hydro simulations, the
#         primitive variable matrix should have five values for each cell
#         (density, x-velocity, y-velocity, z-velocity, and hydrodynamic pressure).
#         For MHD simulations, the primitive variable matrix should have
#         eight values for each cell (density, x-velocity, y-velocity,
#         z-velocity, Bx, By, Bz, and hydrodynamic pressure)
#Input Parameters: primVars (the primitive variables matrix to check)
#                  isMHD (boolean for whether the simulation has
#                         magnetic fields)
#Outputs: formattedPrimVars (the formatted primitive variables matrix)
def formatPrimVars(primVars,isMHD):
    #Get the number of variables in primVars
    numVars = primVars.shape[0]
    #Get the number of spatial dimensions
    numDim = len(primVars.shape) - 1
    #For 1D hydro simulations, the user may pass primitive variable
    #vectors that only have three variables (density, x-velocity,
    #and hydrodynamic pressure). In this scenario, we will add
    #zeros for the y-velocities and z-velocities to make sure
    #that each vector has five variables (as we expect for hydro
    #simulations).
    if numVars == 3 and not isMHD and numDim == 1:
        formattedPrimVars = np.zeros(shape=(5,primVars.shape[1]))
        formattedPrimVars[0,:] = primVars[0,:]
        formattedPrimVars[1,:] = primVars[1,:]
        formattedPrimVars[4,:] = primVars[2,:]
    #For 1D or 2D hydro simulations, the user is allowed
    #to leave out the z-velocities (which will be
    #set to zero by default).
    elif numVars == 4 and not isMHD and (numDim == 1 or numDim == 2):
        formattedPrimVars = np.zeros(shape=(5,primVars.shape[1]))
        formattedPrimVars[0,:] = primVars[0,:]
        formattedPrimVars[1,:] = primVars[1,:]
        formattedPrimVars[2,:] = primVars[2,:]
        formattedPrimVars[4,:] = primVars[3,:]
    #If the simulation doesn't have any magnetic fields and the
    #user passed five variables for each vector, we don't need
    #to change anything.
    elif numVars == 5 and not isMHD:
        formattedPrimVars = np.copy(primVars)
    #Raise a RuntimeError if the user has more than five
    #variables in each cell for a hydro simulation
    elif numVars > 5 and not isMHD:
        raise RuntimeError("Y\nou cannot have more than five variables (dens, vx, vy, vz, pres) for each cell"
                           " in a hydrodynamic simulation. \nIf you want to include magnetic fields, please"
                           " set " + constants.IS_MHD + " = true in your simulation parameters dictionary.")
    #Raise a RuntimeError if the user has fewer than three
    #variables in each cell for a 1D hydro simulation
    elif numVars < 3 and not isMHD and numDim == 1:
        raise RuntimeError("\nYou need at least three variables (dens, vx, pres) for each cell"
                           + " in a 1D hydrodynamic simulation.")
    #Raise a RuntimeError if the user has fewer than four
    #variables in each cell for a 2D hydro simulation
    elif numVars < 4 and not isMHD and numDim == 2:
        raise RuntimeError("\nYou need at least four variables (dens, vx, vy, pres) for each cell"
                           + " in a 2D hydrodynamic simulation.")
    #Raise a RuntimeError if the user has fewer than five
    #variables in each cell for a 3D hydro simulation
    elif numVars < 5 and not isMHD and numDim == 3:
        raise RuntimeError("\nYou need at least five variables (dens, vx, vy, vz, pres) for each cell"
                           + " in a 3D hydrodynamic simulation.")
    #For MHD simulations, we require the user to always have eight variables per cell in their
    #simulation grid. We raise a RuntimeError if their number of variables is larger or smaller
    #than eight.
    elif numVars > 8 and isMHD:
        raise RuntimeError("\nYou provided more than eight variables for the cells in your MHD simulation.\nFor an MHD "
                           + "simulation, you must define exactly eight variables (dens, vx, vy, vz, Bx, By, Bz, pres) "
                           + "in each cell.")
    elif numVars < 8 and isMHD:
        raise RuntimeError("\nYou provided fewer than eight variables for the cells in your MHD simulation. For an MHD "
                           + "simulation,\nyou must define exactly eight variables (dens, vx, vy, vz, Bx, By, Bz, pres) "
                           + "in each cell. \n\nIf you don't want to include magnetic fields, please set " + constants.IS_MHD + " = False "
                           + "in your simulation parameters dictionary\n(or remove the isMHD key, "
                           + "since " + constants.IS_MHD + " = False is the default setting (as long as you haven't changed\n"
                           + " DEFAULT_IS_MHD in PythonMHD_Constants.py)).")
    #If the MHD primVars matrix has eight variables per cell, we don't need
    #to do any formatting.
    else:
        formattedPrimVars = np.copy(primVars)

    return formattedPrimVars


#Function: validatePrimVars
#Purpose: Checks whether the primitive variables matrix has physically valid inputs.
#         The primitive variable matrix will fail the validation test if it has
#         negative densities and/or pressures or if it has a magnetic field with
#         a non-zero divergence (notwithstanding reasonable numerical error).
#         The maximum magnetic field divergence is ________
#Input Parameters: primVars (the primitive variables matrix)
#Outputs: void (this function will raise a RuntimeError if primVars
#               fails the validation test)
def validatePrimVars(primVars, gamma, minDens, minPres, minEnergy):
    #Set the number of variables that we need to compare with minimum values
    #(3, because we need to make sure that density, pressure, and energy are
    #    sufficiently large in each cell)
    numVarsToCheck = 3
    #Get the number of dimensions in primVars
    numDim = len(primVars.shape) - 1
    #Check if the simulation has magnetic fields
    isMHD = primVars.shape[0] == 8
    #Calculate the energy in every cell of the simulation grid
    energy = 0.5*primVars[0]*(primVars[1]*primVars[1]+primVars[2]*primVars[2]+primVars[3]*primVars[3])
    if isMHD:
        energy += primVars[7]/(gamma-1.0)
        energy += 0.5*(primVars[4]*primVars[4]+primVars[5]*primVars[5]+primVars[6]*primVars[6])
    else:
        energy += primVars[4]/(gamma-1.0)
    #Now we will iterate over the three variables to make sure that all densities, pressures, and energies
    #are larger than or equal to their minimum values.
    minVal = minDens #first we'll check the density
    var = primVars[0] #set the current variable to density
    varString = constants.DENSITY #set the string we will show the user to tell them which physical
                          #quantity is causing the problem
    primitiveString = constants.PRIMITIVE #set the string we will show when the problem is in the primitive variables matrix
                                  #(i.e., when densities and pressures are too low)
    conservativeString = constants.CONSERVATIVE #set the string we will show when the problem is in the conservative variables
                                        #matrix (i.e., when energies are too low)
    varTypeString = primitiveString #since we start with density, we should initially use the primitive string
                                    #in any error message
    #Iterate over the three variables we need to check
    for i in range(numVarsToCheck):
        #If we are not on the first cycle (i.e., the one where we check density),
        #we are checking either pressure or energy.
        if i > 0:
            #If we are on the second cycle, we're checking pressure
            if i == 1:
                #Where pressure appears in the primitive variable matrix depends on whether the simulation
                #has magnetic fields
                if isMHD:
                    varIndex = 7
                else:
                    varIndex = 4
                #Use the minimum pressure value
                minVal = minPres
                #Change the variable string
                varString = constants.PRESSURE
                #Get the pressure values that the user has provided
                var = primVars[varIndex]
            else:
                #If we are on the third cycle, we're checking energy
                minVal = minEnergy
                #Change the variable string
                varString = constants.ENERGY
                #Change the variable type string to conservative
                varTypeString = conservativeString
                #Get the energy values we calculated
                var = energy
        #Make sure the variable we are currently checking is greater than zero everywhere
        if np.any(var < 0):
            #Tell the user where the negative values are (up to the maximum number set in PythonMHD_Constants.py).
            #If you want to increase the maximum number of cell locations that PythonMHD will print out, change
            #the MAX_ERROR_CELLS_TO_PRINT constant in PythonMHD_Constants.py.
            errorMessage = "You have at least one negative " + varString + " value in your " + varTypeString\
                            + " variables matrix.\n The negative " + varString + " values are located at the following"\
                            + " indices (up to the first " + str(constants.MAX_ERROR_CELLS_TO_PRINT) + " negative values):\n"
            #Find all the places where the variable is negative
            negValIndices = np.where(var < 0)
            #Get the total number of negative values
            numNegVals = negValIndices[0].shape[0]
            #Determine whether we can print all of the negative cells or just the first MAX_ERROR_CELLS_TO_PRINT
            numIndicesToPrint = np.minimum(numNegVals,constants.MAX_ERROR_CELLS_TO_PRINT)
            #The number of indices/coordinates we need to print out for each cell depends on whether the
            #simulation is 1D, 2D, or 3D
            if numDim == 1:
                for i in range(numIndicesToPrint):
                    errorMessage += " (x_index = " + str(negValIndices[0][i]) + ")"
                    if i < numIndicesToPrint - 1:
                        errorMessage += ","
            elif len(primVars.shape) - 1 == 2:
                for i in range(numIndicesToPrint):
                    errorMessage += " (y_index = " + str(negValIndices[0][i]) \
                                       + ", x_index = " + str(negValIndices[1][i]) + ")"
                    if i < numIndicesToPrint - 1:
                        errorMessage += ","
            else:
                for i in range(numIndicesToPrint):
                    errorMessage += " (y_index = " + str(negValIndices[0][i]) \
                                       + ", x_index = " + str(negValIndices[1][i])\
                                       + ", z_index = " + str(negValIndices[2][i]) + ")"
                    if i < numIndicesToPrint - 1:
                        errorMessage += ","
            #Raise a RuntimeError with the error message we created
            raise RuntimeError(errorMessage)
        elif np.any(var < minVal):
            #Now we check if the variable is greater than zero but less than its minimum value in any of the cells
            errorMessage = "\nYou have at least one " + varString + " value in your " + varTypeString + " variables matrix"\
                            + " that is less than \nthe minimum " + varString + " value for the simulation ("\
                            + str(minVal) + "). The " + varString + " values that fall below this minimum\n"\
                            + "are located at the following indices (up to the first " + str(constants.MAX_ERROR_CELLS_TO_PRINT)\
                            + " too low values):\n"
            #Print out the locations of the cells where the variable is less than the minimum value
            #(up to the MAX_ERROR_CELLS_TO_PRINT constant)
            tooLowValIndices = np.where(var < minVal)
            numLowValues = tooLowValIndices[0].shape[0]
            numIndicesToPrint = np.minimum(numLowValues,constants.MAX_ERROR_CELLS_TO_PRINT)
            if numDim == 1:
                for i in range(numIndicesToPrint):
                    errorMessage += "(x_index = " + str(tooLowValIndices[0][i]) + ")"
                    if i < numIndicesToPrint - 1:
                        errorMessage += ",\n"
            elif numDim == 2:
                for i in range(numIndicesToPrint):
                    errorMessage += "(y_index = " + str(tooLowValIndices[0][i]) \
                                       + ", x_index = " + str(tooLowValIndices[1][i]) + ")"
                    if i < numIndicesToPrint - 1:
                        errorMessage += ",\n"
            else:
                for i in range(numIndicesToPrint):
                    errorMessage += "(y_index = " + str(tooLowValIndices[0][i]) \
                                       + ", x_index = " + str(tooLowValIndices[1][i])\
                                       + ", z_index = " + str(tooLowValIndices[2][i]) + ")"
                    if i < numIndicesToPrint - 1:
                        errorMessage += ",\n"
            #Raise a RuntimeError with the error message we created
            raise RuntimeError(errorMessage)
    #add code for divergence-free condition
    return





#Function: createCoordinates1D_Cartesian
#Purpose: Creates Cartesian coordinate matrix for a 1D simulation based on the
#         parameters in gridPar.
#Input Parameters: gridPar (the grid parameters dictionary that should
#                           be used for constructing the coordinate
#                           matrices)
#Outputs: xCoords (a matrix with the x-coordinates of every cell
#                  in the simulation grid)
def createCoordinates1D_Cartesian(gridPar):
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
    return xCoords, yCoords

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
    return xCoords, yCoords,zCoords

