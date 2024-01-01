#BrioWuShockTube_1D_InitializationFile.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Initialization script for the Brio-Wu shock tube (a classic problem in 1D magnetohydrodynamics).
#Additional Information: The Brio-Wu shock tube [1] is a common test problem for ensuring that
#                        a 1D magnetohydrodynamic simulation accurately propagates MHD waves.
#                        In the Brio-Wu shock tube, we initially have two uniform regions that are separated by
#                        a diaphragm that is removed at time t = 0 (the start of the simulation).
#                        |                       |                        |
#                        |                       |                        |
#                        |         dens_L        |         dens_R         |
#                        |          vx_L         |          vx_R          |
#                        |         pres_L        |         pres_R         |
#                        |          By_L         |          By_R          |
#                        |                       |                        |
#                        |                       |                        |
#                               Left Chamber             Right Chamber
#                        We create the MHD waves by setting the density and pressure of the gas in the
#                        left chamber to be greater than the density and pressure in the right chamber (i.e.,
#                        dens_L > dens_R and pres_L > pres_R). We additionally have the y-component of the magnetic field
#                        flip signs between the two chambers (By_L = +1, By_R = -1). To satisfy the divergence-free
#                        condition on the magnetic field, the x-component of the magnetic field has the same value in
#                        both chambers (Bx = 0.75). The standard values for the Brio-Wu shock tube problem are
#                        dens_L = 1, vx_L = 0.0, By_L = 1, dens_R = 0.125, vx_R = 0, pres_R = 0.1, By_R = -1, Bx = 0.75.
#                        With this initialization file, you can easily change any of these parameters. The densities,
#                        velocities, pressures, and magnetic field components are set in the brioWuPar structure below.
#References:
# [1] Brio, M., and Wu, C. C. (1988). An upwind differencing scheme for the equations of ideal magnetohydrodynamics.
#     Journal of Computational Physics, 75(2), 400-422.

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
              constants.IS_MHD:True, #boolean flag for whether the simulation has magnetic fields
              constants.GAMMA: 2.0, #specific heat ratio for the ideal gas
              constants.TLIM: 0.1, #time at which the simulation should terminate
              constants.CFL: 0.8, #CFL number for calculating timestep sizes
              constants.MAX_CYCLES:10000, #max number of simulation cycles
              constants.RECONSTRUCT_ORDER:constants.NO_SPATIAL_RECONSTRUCTION, #spatial reconstruction order
              constants.ENTROPY_FIX:False, #boolean flag for whether we should apply an entropy fix to small wavespeeds
              constants.EPSILON:0.5, #smallest wavespeed that will not be affected by the entropy fix
              constants.PLOT_DATA:True, #boolean flag for whether we want to visualize the simulation data
              constants.PLOT_DATA_DT:0.1, #amount of simulation time between visualization outputs
              constants.SAVE_FIGS:True, #boolean flag for whether we should save the visualization outputs
              constants.SAVE_FIGS_FORMAT: constants.IMAGE_FORMAT_PNG, #file type for images of any visualization outputs
              constants.SAVE_FIGS_DPI: 300, #resolution in dots-per-inch/dpi for output images
              constants.MATPLOTLIB_BACKEND: constants.MATPLOTLIB_BACKEND_TKAGG, #matplotlib backend (for visualizations)
              constants.SAVE_DATA:False, #boolean flag for whether we should save the numerical data for the simulation
              constants.SAVE_DATA_DT:0.07, #amount of simulation time between numerical data saves
              constants.SAVE_DATA_FORMAT: constants.DATA_FORMAT_MAT, #file type for numerical data saves
              constants.OUTPUT_FOLDER_NAME: "BrioWuShockTube_1D", #name for the outputs folder
              constants.OUTPUT_FILE_NAME: "BrioWuShockTube", #prefix name for the output files
        }

###VISUALIZATION PARAMETERS###
visPar = {
            constants.FIGURES: #array of Figure structures
            [
                #Figure 1
                {
                   constants.FIGURE_TITLE: "Brio-Wu Shock Tube", #title for the figure
                   constants.PLOTS: #array of plot stuctures to display on this figure
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
                           constants.PLOT_VAR: constants.BY, #gas variable we want to plot (By)
                           constants.PLOT_COLOR: "r" #color for the plot (red)
                       },
                       #Plot 4 (energy)
                       {
                           constants.PLOT_VAR: constants.ENERGY, #gas variable we want to plot (energy)
                           constants.PLOT_COLOR: "m" #color for the plot (magenta)
                       }
                    ]
                }
            ]
         }

###BRIO-WU SHOCK TUBE PARAMETERS###
#This parameter structure allow you to customize
#the densities, pressures, and velocities in
#both halves of the Brio-Wu shock tube.
brioWuPar = {
             "Bx": 0.75, #x-component of the magnetic field in both chambers
             "densL": 1.0, #initial density in the left chamber
             "vxL": 0.0, #initial x-velocity in the left chamber
             "presL": 1.0, #initial pressure in the left chamber
             "ByL": 1.0, #initial y-component of the magnetic field in the left chamber
             "densR": 0.125, #initial density in the right chamber
             "vxR": 0.0, #initial x-velocity in the right chamber
             "presR": 0.1, #initial pressure in the right chamber
             "ByR": -1.0, #initial y-component of the magnetic field in the right chamber
             "x0":0.0, #position of the interface between the two chambers
            }


#Function: getBrioWuPrimVars
#Purpose: Creates the primitive variable matrix for the Brio-Wu shock tube problem,
#         using the parameters in gridPar and brioWuPar.
#Input Parameters: gridPar (the grid parameters structure)
#                  brioWuPar (the Brio Wu Shock Tube parameters structure)
#Outputs: primVars (the primitive variable matrix that has been
#                   constructed based on the specifications in
#                   gridPar and sodPar)
def getBrioWuPrimVars(gridPar, brioWuPar):
    #Get the number of cells in the x-direction
    numXCells = gridPar["numXCells"]
    assert(numXCells > 0)
    #Create a matrix with the x-coordinates for all cells in the simulation grid
    xCoords = initHelper.createCoordinates1D_Cartesian(gridPar)
    #Create the density values
    rho = np.zeros(shape=numXCells)
    rho[xCoords <= brioWuPar["x0"]] = brioWuPar["densL"]
    rho[xCoords > brioWuPar["x0"]] = brioWuPar["densR"]
    #Create the x-velocities
    vx = np.zeros(shape=numXCells)
    vx[xCoords <= brioWuPar["x0"]] = brioWuPar["vxL"]
    vx[xCoords > brioWuPar["x0"]] = brioWuPar["vxR"]
    #Set the y-velocities to zero
    vy = np.zeros(shape=numXCells)
    #Set the z-velocities to zero
    vz = np.zeros(shape=numXCells)
    #Set the Bx magnetic field component
    Bx = brioWuPar["Bx"]*np.ones(shape=numXCells)
    #Create the By magnetic field components
    By = np.zeros(shape=numXCells)
    By[xCoords <= brioWuPar["x0"]] = brioWuPar["ByL"]
    By[xCoords > brioWuPar["x0"]] = brioWuPar["ByR"]
    #Set the Bz component to zero
    Bz = np.zeros(shape=numXCells)
    #Create the pressure values
    pres = np.zeros(shape=numXCells)
    pres[xCoords <= brioWuPar["x0"]] = brioWuPar["presL"]
    pres[xCoords > brioWuPar["x0"]] = brioWuPar["presR"]
    #Create the primitive variables matrix,
    #which will contain the eight magnetohydrodynamic
    #primitive variables in the following
    #order: density, x-velocity, y-velocity,
    #       z-velocity, Bx, By, Bz, and pressure
    primVars = np.zeros(shape=(8,numXCells))
    primVars[0,:] = rho
    primVars[1,:] = vx
    primVars[2,:] = vy
    primVars[3,:] = vz
    primVars[4,:] = Bx
    primVars[5,:] = By
    primVars[6,:] = Bz
    primVars[7,:] = pres
    #Return the primitive variables matrix
    return primVars


######MAIN SCRIPT######

#Build the primitive variables matrix for
#the initial state of the Brio-Wu shock tube
brioWuPrimVars = getBrioWuPrimVars(gridPar, brioWuPar)

#Create the Simulation object
brioWuSim = Simulation(simPar,gridPar,brioWuPrimVars,visPar)

#Run the simulation
newGrid = brioWuSim.run()


