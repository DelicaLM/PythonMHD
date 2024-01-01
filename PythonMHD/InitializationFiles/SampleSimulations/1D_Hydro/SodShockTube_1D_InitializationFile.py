#SodShockTube_1D_InitializationFile.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Initialization script for Sod's shock tube (a classic problem in 1D hydrodynamics).
#Additional Information: Sod's shock tube [1] is the most common test problem for ensuring that
#                        a 1D hydrodynamic simulation accurately propagates shock and rarefaction waves.
#                        In Sod's shock tube, we initially have two uniform regions that are separated by
#                        a diaphragm that is removed at time t = 0 (the start of the simulation).
#                        |                       |                        |
#                        |                       |                        |
#                        |         dens_L        |         dens_R         |
#                        |          vx_L         |          vx_R          |
#                        |         pres_L        |         pres_R         |
#                        |                       |                        |
#                        |                       |                        |
#                               Left Chamber             Right Chamber
#                        We create shock and rarefaction waves by setting the density and pressure of the gas in the
#                        left chamber to be greater than the density and pressure in the right chamber (i.e.,
#                        dens_L > dens_R and pres_L > pres_R). Setting the x-velocity in the left chamber to a
#                        positive value (which signifies fluid motion to the right) accelerates the mixing of the
#                        two gas chambers after time t = 0. The standard values for Sod's shock tube problem are
#                        dens_L = 1, vx_L = 0.75, dens_R = 0.125, vx_R = 0, pres_R = 0.1. (To compare against Athena's
#                        default version of Sod's shock tube, vx_L is set to 0 instead of 0.75.) With this initialization
#                        file, you can easily change any of these parameters. The densities, velocities, and pressures
#                        are set in the sodPar structure below.
#References:
# [1] Sod, G. (1978). A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws.
#     Journal of Computational Physics, 27(1), 1-31. https://doi.org/10.1016/0021-9991(78)90023-2.

######IMPORT STATEMENTS######

#Import numpy for matrix operations
import numpy as np

#Import the Simulation class from PythonMHD source code
from Source.Simulation import Simulation

#Import the initialization helper module to find the coordinates of every cell in your simulation grid
import Source.Initialization_HelperModule as initHelper

#Import the module that contains all PythonMHD constants (convenient for setting input parameters)
import Source.PythonMHD_Constants as constants

######INPUT STRUCTURES######

###GRID PARAMETERS###
gridPar = {
            constants.NUM_X_CELLS: 1000, #set the number of cells in the x-direction
            constants.MIN_X:-0.5, #set the minimum x-coordinate
            constants.MAX_X:0.5, #set the maximum x-coordinate
            constants.BC_X:constants.OUTFLOW, #set the boundary condition in the x-direction
          }

###SIMULATION PARAMETERS###
simPar = {
              constants.IS_MHD:False, #boolean flag for whether the simulation has magnetic fields
              constants.GAMMA: 1.4, #specific heat ratio for the ideal gas
              constants.TLIM: 0.25, #time at which the simulation should terminate
              constants.CFL: 0.8, #CFL number for calculating timestep sizes
              constants.MAX_CYCLES:10000, #max number of simulation cycles
              constants.RECONSTRUCT_ORDER:constants.PPM_SPATIAL_RECONSTRUCTION, #spatial reconstruction order
              constants.ENTROPY_FIX:False, #boolean flag for whether we should apply an entropy fix to small wavespeeds
              constants.EPSILON:0.5, #smallest wavespeed that will not be affected by the entropy fix
              constants.PLOT_DATA:False, #boolean flag for whether we want to visualize the simulation data
              constants.PLOT_DATA_DT:0.05, #amount of simulation time between visualization outputs
              constants.SAVE_FIGS:False, #boolean flag for whether we should save the visualization outputs
              constants.SAVE_FIGS_FORMAT: constants.IMAGE_FORMAT_PNG, #file type for images of any visualization outputs
              constants.SAVE_FIGS_DPI: 300, #resolution in dots-per-inch/dpi for output images
              constants.MAKE_MOVIE: False, #boolean flag for whether PythonMHD should make a movie of each matplotlib figure
              constants.MATPLOTLIB_BACKEND: constants.MATPLOTLIB_BACKEND_TKAGG, #matplotlib backend (for visualizations)
              constants.SAVE_DATA:False, #boolean flag for whether we should save the numerical data for the simulation
              constants.SAVE_DATA_DT:0.05, #amount of simulation time between numerical data saves
              constants.SAVE_DATA_FORMAT: constants.DATA_FORMAT_MAT, #file type for numerical data saves
              constants.OUTPUT_FOLDER_NAME: "SodShockTube_1D", #name for the outputs folder
              constants.OUTPUT_FILE_NAME: "SodShockTube", #prefix name for the output files
         }

###VISUALIZATION PARAMETERS###
visPar = {
            constants.FIGURES: #array of Figure structures
            [
                #Figure 1
                {
                   constants.FIGURE_TITLE: "Sod's Shock Tube", #title for the figure
                   constants.PLOTS: #array of plots to display on this figure
                    [
                       #Plot 1 (density)
                       {
                           constants.PLOT_VAR: constants.DENSITY, #gas variable we want to plot (density)
                           constants.PLOT_COLOR: "b" #color for the plot (blue)
                       },
                       #Plot 2 (x-velocity)
                       {
                           constants.PLOT_VAR: constants.VX, #gas variable we want to plot (x-velocity)
                           constants.PLOT_COLOR: "c" #color for the plot (cyan)
                       },
                       #Plot 3 (pressure)
                       {
                           constants.PLOT_VAR: constants.PRESSURE, #gas variable we want to plot (pressure)
                           constants.PLOT_COLOR: "r" #color for the plot (red)
                       },
                       #Plot 4 (energy)
                       {
                           constants.PLOT_VAR: constants.ENERGY, #gas variable we want to plot (energy)
                           constants.PLOT_COLOR: "m" #color for the plot (magenta)
                       }
                    ]
                },
                {
                    constants.PLOTS: [{}]
                }
            ]
         }

###SOD SHOCK TUBE PARAMETERS###
#This parameter structure allow you to customize
#the densities, pressures, and velocities in
#both halves of Sod's shock tube.
sodPar = {
          "densL": 1.0, #initial density in the left chamber
          "vxL": 0.0, #initial x-velocity in the left chamber
          "presL": 1.0, #initial pressure in the left chamber
          "densR": 0.125, #initial density in the right chamber
          "vxR": 0.0, #initial x-velocity in the right chamber
          "presR": 0.1, #initial pressure in the right chamber
          "x0":0.0, #position of the interface between the two chambers
          }


######PRIMITIVE VARIABLE FUNCTION######

#Function: getSodPrimVars
#Purpose: Creates the primitive variable matrix for Sod's shock tube problem,
#         using the parameters in gridPar and sodPar.
#Input Parameters: gridPar (the grid parameters structure)
#                  sodPar (the Sod Shock Tube parameters structure)
#Outputs: primVars (the primitive variable matrix that has been
#                   constructed based on the specifications in
#                   gridPar and sodPar)
def getSodPrimVars(gridPar, sodPar):
    #Get the number of cells that we should have in the x-direction
    numXCells = gridPar[constants.NUM_X_CELLS]
    assert(numXCells > 0)
    #Calculate the width of each grid cell
    dx = (gridPar[constants.MAX_X] - gridPar[constants.MIN_X])/numXCells
    assert(dx > 0)
    #Create a matrix with the x-coordinates for all cells in the simulation grid
    xCoords = initHelper.createCoordinates1D_Cartesian(gridPar)
    #Get the position of the interface between the left and right chambers
    x0 = sodPar["x0"]
    assert(gridPar[constants.MIN_X] < x0 < gridPar[constants.MAX_X])
    #Create the density values
    rho = np.zeros(shape=numXCells)
    rho[xCoords <= sodPar["x0"]] = sodPar["densL"]
    rho[xCoords > sodPar["x0"]] = sodPar["densR"]
    #Create the x-velocities
    vx = np.zeros(shape=numXCells)
    vx[xCoords <= sodPar["x0"]] = sodPar["vxL"]
    vx[xCoords > sodPar["x0"]] = sodPar["vxR"]
    #Set the y-velocities to zero
    vy = np.zeros(shape=numXCells)
    #Set the z-velocities to zero
    vz = np.zeros(shape=numXCells)
    #Create the presssure values
    pres = np.zeros(shape=numXCells)
    pres[xCoords <= sodPar["x0"]] = sodPar["presL"]
    pres[xCoords > sodPar["x0"]] = sodPar["presR"]
    #Create the primitive variables matrix,
    #which will contain the five hydrodynamic
    #primitive variables in the following
    #order: density, x-velocity, y-velocity,
    #       z-velocity, and pressure
    primVars = np.zeros(shape=(5,numXCells))
    primVars[0,:] = rho
    primVars[1,:] = vx
    primVars[2,:] = vy
    primVars[3,:] = vz
    primVars[4,:] = pres
    #Return the primitive variables matrix
    return primVars

######MAIN SCRIPT######

#Build the primitive variables matrix for
#the initial state of Sod's shock tube
print("Building Primitive Variable Matrix\n")
sodPrimVars = getSodPrimVars(gridPar, sodPar)

#Create a Simulation object
print("Creating Simulation Object\n")
sodSim = Simulation(simPar,gridPar,sodPrimVars,visPar)

#Run the Simulation
print("Running Simulation\n")
newGrid = sodSim.run()







