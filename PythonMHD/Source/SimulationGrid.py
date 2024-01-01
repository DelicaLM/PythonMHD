#SimulationGrid.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
# Purpose: Defines the SimulationGrid class for PythonMHD simulations. When a user creates a Simulation object, the
#          constructor in the Simulation class will automatically create a SimulationGrid object to store the primitive
#          (density, velocities, magnetic field components, pressure) and conservative (density, momenta, magnetic
#          field components, energy) variables for every cell. In MHD simulations, the SimulationGrid object also stores
#          the face-centred magnetic field values, which are necessary to maintain the divergence-free condition.

##########IMPORT STATEMENTS###########

#Import numpy for matrix operations
import numpy as np

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants

#Import the SimulationGrid Helper Moduler (provides the validation functions for checking the user's
#grid parameters and initial conditions)
import Source.SimulationGrid_HelperModule as simGridLib

#Import the helper module for converting between primitive and conservative variables
import Source.PrimCons_HelperModule as primConsLib

#Import the Constrained Transport helper module for calculating the face-centred magnetic field values
import Source.CT_HelperModule as ctLib


###############SIMULATION GRID CLASS#################
#Class: SimulationGrid
#Purpose: A SimulationGrid object stores critical information (i.e., coordinates for every cell, all cell-centred
#         primitive and conservative variables, and face-centred magnetic fields) about the grid that is being
#         used in a Simulation.
class SimulationGrid:
    #########CONSTRUCTOR METHOD##########
    #Function: __init__
    #Purpose: Creates a new Simulation Grid object (if the user provides valid
    #         inputs for the primitive variables matrix and the grid parameters).
    #Inputs: primVars (the primitive variables matrix)
    #        gridPar (the grid parameters)
    #        isMHD (boolean flag for checking whether the simulation grid
    #               should have magnetic fields)
    #        gamma (the specific heat ratio for the ideal gas
    #               (used for converting primitive variables
    #                into conservative variables))
    #        minDens
    def __init__(self,primVars,gridPar,isMHD,gamma,minDens,minPres,minEnergy,consVars,faceBx,faceBy,faceBz):
        #Get rid of any singleton dimensions in primVars
        primVars = np.squeeze(primVars)

        #Validate the grid parameters (also checks if gridPar agrees
        #with the dimensions of primVars)
        simGridLib.validateGridPar(primVars,gridPar)

        #Check if we are running a simulation with magnetic fields
        self.isMHD = isMHD

        #Ensure that primVars follows the standard format for hydrodynamic
        #and MHD simulations (5 variables per cell for hydro (dens, vx, vy,
        #vz, and pres), 8 variables per cell for MHD (dens, vx, vy, vz, Bx,
        #By, Bz, pres).
        primVars = simGridLib.formatPrimVars(primVars,self.isMHD)

        #Validate the primitive variable values (to make sure that there
        #are no negative densities or pressures)
        simGridLib.validatePrimVars(primVars, gamma, minDens, minPres, minEnergy)

        #Get the number of spatial dimensions in primVars (with singleton dimensions removed)
        self.nDim = len(primVars.shape) - 1

        #Now we can set the primitive variables for the simulation grid
        self.primVars = primVars
        #Use the primitive variables to calculate the conservative variables
        #everywhere on the simulation grid
        if(consVars.shape[0] > 1):
            self.consVars = consVars
        else:
            if self.isMHD:
                self.consVars = primConsLib.primToCons_mhd_allVars(self.primVars, gamma)
            else:
                self.consVars = primConsLib.primToCons_hydro(self.primVars, gamma)

        #Next we need to create the coordinate matrices. This version of PythonMHD only supports Cartesian coordinates.
        #Future versions will include support for cylindrical and spherical coordinate systems.

        #Save the MIN and MAX x, y, and z coordinates
        self.minX = gridPar[constants.MIN_X]
        self.maxX = gridPar[constants.MAX_X]
        self.numXCells = gridPar[constants.NUM_X_CELLS]
        self.numYCells = 1
        self.numZCells = 1
        if self.nDim >= 2:
            self.minY = gridPar[constants.MIN_Y]
            self.maxY = gridPar[constants.MAX_Y]
            self.numYCells = gridPar[constants.NUM_Y_CELLS]
        if self.nDim == 3:
            self.minZ = gridPar[constants.MIN_Z]
            self.maxZ = gridPar[constants.MAX_Z]
            self.numZCells = gridPar[constants.NUM_Z_CELLS]

        #For convenience, we will save the cell size along each required dimension.
        #dx = cell width (size of the cell in the x-direction)
        #dy = cell height (size of the cell in the y-direction)
        #dz = cell depth (size of the cell in the z-direction)
        #For simulations with uniform grids, we will only have one set of (dx, dy, dz).
        #When Adaptive Mesh Refinement (AMR) is introduced, this place in the code
        #would be a convenient place to store the variable sizes of cells (i.e.,
        #dx, dy, and dz would all become matrices instead of single values).

        #Calculate and save the cell width
        self.dx = (self.maxX - self.minX)/self.numXCells
        #If the sim is 2D or 3D, calculate and save the cell height
        if self.nDim >= 2:
            self.dy = (self.maxY - self.minY)/self.numYCells
        #If the sim is 3D, calculate and save the cell depth
        if self.nDim == 3:
            self.dz = (self.maxZ - self.minZ)/self.numZCells

        #Next we call the helper function for getting the coordinate
        #matrices that we will need for the Cartesian simulation
        if self.nDim == 1:
            self.xCoords = simGridLib.createCoordinates1D_Cartesian(gridPar)
        elif self.nDim == 2:
            (self.xCoords, self.yCoords) = simGridLib.createCoordinates2D_Cartesian(gridPar)
        else:
            (self.xCoords, self.yCoords, self.zCoords) = simGridLib.createCoordinates3D_Cartesian(gridPar)

        #Lastly, we will set the boundary conditions for the x-, y-, and z-directions
        #The boundary conditions are set by the user in gridPar with the parameters
        #"BcX", "BcY", and "BcZ". A value of 0 corresponds to an outflow boundary condition,
        #whereas 1 is used for periodic boundary conditions. If the user does not specify
        #a boundary condition in gridPar, the boundary condition will be set to outflow
        #by default. If you want the default boundary condition to be periodic instead,
        #you can change DEFAULT_BC_X, DEFAULT_BC_Y, and/or DEFAULT_BC_Z in PythonMHD_Constants.py
        self.BcX = constants.DEFAULT_BC_X
        self.BcY = constants.DEFAULT_BC_Y
        self.BcZ = constants.DEFAULT_BC_Z
        if constants.BC_X in gridPar.keys():
            if gridPar[constants.BC_X] == constants.PERIODIC:
                self.BcX = constants.PERIODIC
        if constants.BC_Y in gridPar.keys():
            if gridPar[constants.BC_Y] == constants.PERIODIC:
                self.BcY = constants.PERIODIC
        if constants.BC_Z in gridPar.keys():
            if gridPar[constants.BC_Z] == constants.PERIODIC:
                self.BcZ = constants.PERIODIC
        if self.isMHD:
            if self.nDim == 2:
                if faceBx.shape[0] > 1:
                    self.faceBx = faceBx
                else:
                    self.faceBx = ctLib.getFaceBx_2D(self.primVars[4],self.BcX)
                if faceBy.shape[0] > 1:
                    self.faceBy = faceBy
                else:
                    self.faceBy = ctLib.getFaceBy_2D(self.primVars[5],self.BcY)
            elif self.nDim == 3:
                if faceBx.shape[0] > 1:
                    self.faceBx = faceBx
                else:
                    self.faceBx = ctLib.getFaceBx_3D(self.primVars[4],self.BcX)
                if faceBy.shape[0] > 1:
                    self.faceBy = faceBy
                else:
                    self.faceBy = ctLib.getFaceBy_3D(self.primVars[5],self.BcY)
                if faceBz.shape[0] > 1:
                    self.faceBz = faceBz
                else:
                    self.faceBz = ctLib.getFaceBz_3D(self.primVars[6],self.BcZ)



