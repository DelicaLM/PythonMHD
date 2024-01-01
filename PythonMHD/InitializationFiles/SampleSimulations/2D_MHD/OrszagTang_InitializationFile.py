#OrszagTang_InitializationFile.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Initialization script for Orszag-Tang vortex (a standard test problem for 2D magnetohydrodynamics).
#Additional Information: This problem is the vortex proposed in [1]. This initialization script
#                        was designed to produce the same gas system as the Orszag-Tang vortex problem in the 2017
#                        version of Athena [2][3]. We create the vortex by having the initial velocities and magnetic
#                        field components vary sinusoidally across a square of gas that initially has a uniform density
#                        and pressure.
#References:
# 1. Orszag, S. A., & Tang, C. M. (1979). Small-scale structure of two-dimensional magnetohydrodynamic turbulence.
#    Journal of Fluid Mechanics, 90(1). 129-143. https://doi.org/10.1017/S002211207900210X.
# 2. Stone, J. M., Gardiner, T. A., Teuben, P., Hawley, J. F., & Simon, J. B. (2008).
#    Athena: A new code for astrophysical MHD. The Astrophysical Journal Supplemental Series,
#    178(1), 137-177. https://iopscience.iop.org/article/10.1086/588755/pdf.
# 3. https://github.com/PrincetonUniversity/Athena-Cversion

######IMPORT STATEMENTS######

#Import numpy for matrix operations
import numpy as np

#Import the Simulation class from PythonMHD source code
from Source.Simulation import Simulation

#Import the initialization helper module to find the coordinates of every cell in your simulation grid
import Source.Initialization_HelperModule as initHelper

#Import the module that contains all PythonMHD constants (convenient for setting input parameters)
import Source.PythonMHD_Constants as constants

#Import math so we can use PI to calculate angle values in radians
import math

PI = math.pi


######INPUT STRUCTURES######

###GRID PARAMETERS###
gridPar = {
            constants.NUM_X_CELLS: 128, #set the number of cells in the x-direction
            constants.MIN_X:0.0, #set the minimum x-coordinate
            constants.MAX_X:1.0, #set the maximum x-coordinate
            constants.NUM_Y_CELLS: 128, #set the number of cells in the y-direction
            constants.MIN_Y:0.0, #set the minimum y-coordinate
            constants.MAX_Y:1.0, #set the maximum y-coordinate
            constants.BC_X:constants.PERIODIC, #set the boundary condition in the x-direction
            constants.BC_Y:constants.PERIODIC, #set the boundary condition in the y-direction
          }

###SIMULATION PARAMETERS###
simPar = {
              constants.IS_MHD:True, #boolean flag for whether the simulation has magnetic fields
              constants.GAMMA: 1.666666667, #specific heat ratio for the ideal gas
              constants.TLIM: 1.0, #time at which the simulation should terminate
              constants.CFL: 0.8, #CFL number for calculating timestep sizes
              constants.MAX_CYCLES:100000, #max number of simulation cycles
              constants.RECONSTRUCT_ORDER:constants.PPM_SPATIAL_RECONSTRUCTION, #spatial reconstruction order
              constants.ENTROPY_FIX:False, #boolean flag for whether we should apply an entropy fix to small wavespeeds
              constants.EPSILON:0.5, #smallest wavespeed that will not be affected by the entropy fix
              constants.PLOT_DATA:False, #boolean flag for whether we want to visualize the simulation data
              constants.PLOT_DATA_DT:0.05, #amount of simulation time between visualization outputs
              constants.SAVE_FIGS:False, #boolean flag for whether we should save the visualization outputs
              constants.SAVE_FIGS_FORMAT: constants.IMAGE_FORMAT_PNG, #file type for images of any visualization outputs
              constants.SAVE_FIGS_DPI: 300, #resolution in dots-per-inch/dpi for output images
              constants.MATPLOTLIB_BACKEND: constants.MATPLOTLIB_BACKEND_TKAGG, #matplotlib backend (for visualizations)
              constants.SAVE_DATA:False, #boolean flag for whether we should save the numerical data for the simulation
              constants.SAVE_DATA_DT:0.05, #amount of simulation time between numerical data saves
              constants.SAVE_DATA_FORMAT: constants.DATA_FORMAT_MAT, #file type for numerical data saves
              constants.OUTPUT_FOLDER_NAME: "OrszagTangVortex", #name for the outputs folder
              constants.OUTPUT_FILE_NAME: "OrszagTangVortex", #prefix name for the output files
         }


###VISUALIZATION PARAMETERS###
visPar = {
            constants.FIGURES: #array of Figure structures
            [
                #Figure 1
                {
                   constants.FIGURE_TITLE: "Orszag-Tang Vortex", #title for the figure
                   constants.PLOTS: #array of plot stuctures to display on this figure
                    [
                       {
                           constants.PLOT_VAR: constants.PRESSURE, #gas variable to plot (pressure)
                           constants.PLOT_COLOR: "viridis", #colormap for the 2D plot
                           constants.MIN_PLOT_VAL: 0.01, #minimum value on the colormap scale
                           constants.MAX_PLOT_VAL: 0.7, #maximum value on the colormap scale
                       },
                       {
                            constants.PLOT_VAR: constants.B, #gas variable to plot (magnetic field strength)
                            constants.PLOT_COLOR: "plasma", #colormap for the 2D plot
                       },
                       {
                           constants.PLOT_VAR: constants.PRESSURE, #gas variable to plot (pressure)
                           constants.PLOT_TYPE: constants.PLOT_TYPE_1D, #type of plot (1D line plot)
                           constants.PLOT_AXIS: constants.Y_AXIS, #axis parallel with the line (x-axis)
                           constants.X_POS: 0.0, #y-position of the line (0.0)
                       },
                       {
                           constants.PLOT_VAR: constants.B, #gas variable to plot (magnetic field strength)
                           constants.PLOT_TYPE: constants.PLOT_TYPE_1D, #type of plot (1D line plot)
                           constants.PLOT_AXIS: constants.X_AXIS, #axis parallel with the line (x-axis)
                           constants.Y_POS: 0.0, #y-position of the line (0.0)
                       }
                    ]
                }
            ]
         }

###ORSZAG-TANG VORTEX PARAMETERS###
#This parameter structure allow you to customize
#the initial density, pressure, x-velocity scaling factor,
#y-velocity scaling factor, and magnetic field strength
#in the Orszag-Tang vortex.
oztPar = {"density":(25.0/(36.0*PI)), #initial uniform density
            "pressure": (5.0/(12.0*PI)), #initial uniform pressure
            "v_0x":1.0, #scaling factor for initial velocities in the x-direction
            "v_0y":1.0, #scaling factor for initial velocities in the y-direction
            "B_0":1.0/np.sqrt(4.0*PI)} #scaling factor for the magnetic field components


######PRIMITIVE VARIABLE FUNCTION######

#Function: getOZTPrimVars
#Purpose: Creates the primitive variable matrix for the Orszag-Tang vortex problem,
#         using the parameters in gridPar and sodPar.
#Input Parameters: gridPar (the grid parameters structure)
#                  oztPar (the Orszag-Tang vortex parameters structure)
#Outputs: primVars (the primitive variable matrix that has been
#                   constructed based on the specifications in
#                   gridPar and oztPar)
def getOZTPrimVars(gridPar, oztPar):
    #Get the number of cells that we should have in the x-direction
    numXCells = gridPar[constants.NUM_X_CELLS]
    assert(numXCells > 0)
    #Get the number of cells that we should have in the y-direction
    numYCells = gridPar[constants.NUM_Y_CELLS]
    assert(numYCells > 0)
    #Get the width of each cell
    dx = (gridPar[constants.MAX_X] - gridPar[constants.MIN_X])/gridPar[constants.NUM_X_CELLS]
    #Get the height of each cell
    dy = (gridPar[constants.MAX_Y] - gridPar[constants.MIN_Y])/gridPar[constants.NUM_Y_CELLS]
    #Get the x- and y-coordinates for all cells in the simulation grid
    (xCoords, yCoords) = initHelper.createCoordinates2D_Cartesian(gridPar)

    #Calculate the magnetic potential at cell edges
    potential = (oztPar["B_0"]/(4.0*PI))*np.cos(4.0*PI*(xCoords-dx/2.0)) \
                + (oztPar["B_0"]/(2.0*PI))*np.cos(2.0*PI*(yCoords-dy/2.0))

    #Create the density values
    rho = oztPar["density"]*np.ones(shape=(gridPar[constants.NUM_Y_CELLS], gridPar[constants.NUM_X_CELLS]))
    #Create the pressure values
    pres = oztPar["pressure"] * np.ones(shape=(gridPar[constants.NUM_Y_CELLS], gridPar[constants.NUM_X_CELLS]))
    #Create the x-velocities
    vx = oztPar["v_0x"]*(-np.sin(2.0*PI*yCoords))
    #Create the y-velocities
    vy = oztPar["v_0y"]*(np.sin(2.0*PI*xCoords))
    #Set the z-velocities to zero
    vz = np.zeros(shape=xCoords.shape)

    #Calculate the face-centred Bx values from the magnetic potential
    leftBx = (np.append(potential[1:potential.shape[0],:], potential[0,:].reshape(1,-1), axis=0) - potential)/dy
    rightBx = np.append(leftBx[:,1:leftBx.shape[1]], leftBx[:,0].reshape(-1,1), axis=1)
    #Calculate the cell-centred Bx values
    centBx = (leftBx + rightBx) / 2.0

    #Calculate the face-centred By values from the magnetic potential
    topBy = -1.0*(np.append(potential[:,1:potential.shape[1]], potential[:,0].reshape(-1,1), axis=1) - potential)/dx
    bottomBy = np.append(topBy[1:topBy.shape[0],:], topBy[0,:].reshape(1,-1), axis=0)
    #Calculate the cell-centred By values
    centBy = (topBy + bottomBy) / 2.0

    #Set the Bz components to zero
    centBz = np.zeros(shape=(gridPar[constants.NUM_Y_CELLS], gridPar[constants.NUM_X_CELLS]))

    #Create the primitive variables matrix,
    #which will contain the eight magnetohydrodynamic
    #primitive variables in the following
    #order: density, x-velocity, y-velocity,
    #       z-velocity, Bx, By, Bz, and pressure
    primVars = np.zeros(shape=(8,gridPar[constants.NUM_Y_CELLS],gridPar[constants.NUM_X_CELLS]))
    primVars[0,:] = rho
    primVars[1,:] = vx
    primVars[2,:] = vy
    primVars[3,:] = vz
    primVars[4,:] = centBx
    primVars[5,:] = centBy
    primVars[6,:] = centBz
    primVars[7,:] = pres

    #Return the primitive variables matrix
    return primVars


######MAIN SCRIPT######

#Build the primitive variables matrix for
#the initial state of the Orszag-Tang vortex
oztPrimVars = getOZTPrimVars(gridPar, oztPar)

#Create the Simulation object
oztSim = Simulation(simPar,gridPar,oztPrimVars,visPar)

#Run the simulation
newGrid = oztSim.run()

