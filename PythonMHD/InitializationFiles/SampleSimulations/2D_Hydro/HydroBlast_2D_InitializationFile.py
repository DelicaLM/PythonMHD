#HydroBlast_2D_InitializationFile.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Initialization script for 2D hydrodynamic blast (a standard test problem for 2D hydrodynamics).
#Additional Information: This problem is the hydrodynamic version of the MHD blast in [1]. This initialization script
#                        was designed to produce the same gas system as the 2D hydrodynamic blast in the 2017 version
#                        of Athena [2][3]. We create the blast by placing a high-pressure disk of gas at the centre
#                        of the simulation grid. As long as the disk gas has a higher pressure than the ambient medium,
#                        it will expand into the rest of the simulation volume.
#References:
# 1. Gardiner, T. A., & Stone, J. M. (2005). An unsplit Godunov method for ideal MHD via Constrained Transport.
#    Journal of Computational Physics, 205(2), 509-539. https://arxiv.org/pdf/astro-ph/0501557.pdf.
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

######INPUT STRUCTURES######

###GRID PARAMETERS###
gridPar = {
            constants.NUM_X_CELLS: 200, #set the number of cells in the x-direction
            constants.MIN_X:-0.5, #set the minimum x-coordinate
            constants.MAX_X:0.5, #set the maximum x-coordinate
            constants.NUM_Y_CELLS: 300, #set the number of cells in the y-direction
            constants.MIN_Y:-0.75, #set the minimum y-coordinate
            constants.MAX_Y:0.75, #set the maximum y-coordinate
            constants.BC_X:constants.PERIODIC, #set the boundary condition in the x-direction
            constants.BC_Y:constants.PERIODIC, #set the boundary condition in the y-direction
          }

###SIMULATION PARAMETERS###
simPar = {
              constants.IS_MHD:False, #boolean flag for whether the simulation has magnetic fields
              constants.GAMMA: 1.6666666667, #specific heat ratio for the ideal gas
              constants.TLIM: 1.0, #time at which the simulation should terminate
              constants.CFL: 0.8, #CFL number for calculating timestep sizes
              constants.MAX_CYCLES:10000, #max number of simulation cycles
              constants.RECONSTRUCT_ORDER:constants.NO_SPATIAL_RECONSTRUCTION, #spatial reconstruction order
              constants.ENTROPY_FIX:False, #boolean flag for whether we should apply an entropy fix to small wavespeeds
              constants.EPSILON:0.5, #smallest wavespeed that will not be affected by the entropy fix
              constants.PLOT_DATA:False, #boolean flag for whether we want to visualize the simulation data
              constants.PLOT_DATA_DT:0.1, #amount of simulation time between visualization outputs
              constants.SAVE_FIGS:False, #boolean flag for whether we should save the visualization outputs
              constants.SAVE_FIGS_FORMAT: constants.IMAGE_FORMAT_PNG, #file type for images of any visualization outputs
              constants.SAVE_FIGS_DPI: 300, #resolution in dots-per-inch/dpi for output images
              constants.MATPLOTLIB_BACKEND: constants.MATPLOTLIB_BACKEND_TKAGG, #matplotlib backend (for visualizations)
              constants.SAVE_DATA:False, #boolean flag for whether we should save the numerical data for the simulation
              constants.SAVE_DATA_DT:0.1, #amount of simulation time between numerical data saves
              constants.SAVE_DATA_FORMAT: constants.DATA_FORMAT_MAT, #file type for numerical data saves
              constants.OUTPUT_FOLDER_NAME: "HydroBlast_2D", #name for the outputs folder
              constants.OUTPUT_FILE_NAME: "HydroBlast_2D", #prefix name for the output files
         }

###VISUALIZATION PARAMETERS###
visPar = {
            constants.FIGURES: #array of Figure structures
            [
                #Figure 1
                {
                   constants.FIGURE_TITLE: "2D Hydro Blast", #title for the figure
                   constants.PLOTS: #array of plot stuctures to display on this figure
                    [
                        #Plot 1 (2D density plot)
                       {
                           constants.PLOT_VAR: constants.DENSITY, #gas variable to plot (density)
                           constants.PLOT_COLOR: "viridis" #colormap for the 2D plot
                       },
                       #Plot 2 (2D pressure plot)
                       {
                            constants.PLOT_VAR: constants.PRESSURE, #gas variable to plot (pressure)
                            constants.PLOT_COLOR: "plasma", #colormap for the 2D plot
                       },
                       #Plot 3 (1D density plot at y = 0)
                       {
                           constants.PLOT_VAR: constants.DENSITY, #gas variable to plot (density)
                           constants.PLOT_TYPE: constants.PLOT_TYPE_1D, #type of plot (1D line plot)
                           constants.PLOT_AXIS: constants.X_AXIS, #axis parallel with the line (x-axis)
                           constants.Y_POS: 0.0, #y-position of the line (0.0)
                       },
                       #Plot 4 (1D pressure plot at y = 0)
                       {
                           constants.PLOT_VAR: constants.PRESSURE, #gas variable to plot (pressure)
                           constants.PLOT_TYPE: constants.PLOT_TYPE_1D, #type of plot (1D line plot)
                           constants.PLOT_AXIS: constants.X_AXIS, #axis parallel with the line (x-axis)
                           constants.Y_POS: 0.0, #y-position of the line (0.0)
                       },
                    ]
                }
            ]
         }

###HYDRO BLAST PARAMETERS###
#This parameter structure allow you to customize
#ambient pressure, blast pressure, ambient density,
#blast density, and blast radius for the 2D hydrodynamic blast.
blastPar = {
            "ambPres":0.1, #initial pressure in the ambient medium/area outside the blast
            "presRatio":100.0, #initial pressure ratio between the blast and the ambient medium
                               #(blast pressure/ambient pressure)
            "ambDens":1.0, #initial density in the ambient medium/area outside the blast
            "densRatio":1.0, #initial density ratio between the blast and the ambient medium
                             #(blast density/ambient density)
            "blastRadius":0.1 #initial size of the blast/over-pressurized region
           }


######PRIMITIVE VARIABLE FUNCTION######

#Function: getBlastPrimVars
#Purpose: Creates the primitive variable matrix for the 2D hydro blast,
#         using the parameters in gridPar and blastPar.
#Input Parameters: gridPar (the grid parameters structure)
#                  blastPar (the Sod Shock Tube parameters structure)
#Outputs: primVars (the primitive variable matrix that has been
#                   constructed based on the specifications in
#                   gridPar and blastPar)
def getBlastPrimVars(gridPar, blastPar):
    #Get the number of cells that we should have in the x-direction
    numXCells = gridPar[constants.NUM_X_CELLS]
    assert(numXCells > 0)
    #Get the number of cells that we should have in the y-direction
    numYCells = gridPar[constants.NUM_Y_CELLS]
    assert (numYCells > 0)
    #Create the x- and y-coordinates for the simulation
    (xCoords, yCoords) = initHelper.createCoordinates2D_Cartesian(gridPar)
    #Calculate each cell's distance from the centre of the simulation grid (at coordinates (x = 0, y = 0))
    radii = np.sqrt(xCoords*xCoords + yCoords*yCoords)

    #Get the blast parameters
    blastRadius = blastPar["blastRadius"]
    ambPres = blastPar["ambPres"]
    presRatio = blastPar["presRatio"]
    ambDens = blastPar["ambDens"]
    densRatio = blastPar["densRatio"]

    #Create the density values
    rho = ambDens*np.ones(shape=(numYCells,numXCells))
    rho[radii < blastRadius] = densRatio*ambDens
    #Set the x-velocities to zero
    vx = np.zeros(shape=(numYCells,numXCells))
    #Set the y-velocities to zero
    vy = np.zeros(shape=numXCells)
    #Set the z-velocities to zero
    vz = np.zeros(shape=numXCells)
    #Create the pressure values
    pres = ambPres*np.ones(shape=(numYCells,numXCells))
    pres[radii < blastRadius] = presRatio*ambPres
    #Create the primitive variables matrix,
    #which will contain the five hydrodynamic
    #primitive variables in the following
    #order: density, x-velocity, y-velocity,
    #       z-velocity, and pressure
    primVars = np.zeros(shape=(5,numYCells,numXCells))
    primVars[0,:] = rho
    primVars[1,:] = vx
    primVars[2,:] = vy
    primVars[3,:] = vz
    primVars[4,:] = pres
    #Return the primitive variables matrix
    return primVars

######MAIN SCRIPT######

#Build the primitive variables matrix for
#the initial state of the 2D hydro blast
blastPrimVars = getBlastPrimVars(gridPar, blastPar)

#Create the Simulation object
blastSim = Simulation(simPar,gridPar,blastPrimVars,visPar)

#Run the simulation
newGrid = blastSim.run()

