#HydroBlast_3D_InitializationFile.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Initialization script for 3D hydrodynamic blast (a standard test problem for 3D hydrodynamics).
#Additional Information: This problem is the hydrodynamic version of the MHD blast in [1]. This initialization script
#                        was designed to produce the same gas system as the 3D hydrodynamic blast in the 2017 version
#                        of Athena [2][3]. We create the blast by placing a high-pressure sphere of gas at the centre
#                        of the simulation grid. As long as the sphere gas has a higher pressure than the ambient medium,
#                        it will expand into the rest of the simulation volume.
#References:
# 1. Gardiner, T. A., & Stone, J. M. (2005). An unsplit Godunov method for ideal MHD via Constrained Transport.
#    Journal of Computational Physics, 205(2), 509-539. https://arxiv.org/pdf/astro-ph/0501557.pdf.
# 2. Stone, J. M., Gardiner, T. A., Teuben, P., Hawley, J. F., & Simon, J. B. (2008).
#    Athena: A new code for astrophysical MHD. The Astrophysical Journal Supplemental Series,
#    178(1), 137-177. https://iopscience.iop.org/article/10.1086/588755/pdf.
# 3. https://github.com/PrincetonUniversity/Athena-Cversion

######IMPORT STATEMENTS######

#Import the Simulation class from PythonMHD source code
from Source.Simulation import Simulation

#Import the initialization helper module to find the coordinates of every cell in your simulation grid
import Source.Initialization_HelperModule as initHelper

#Import the module that contains all PythonMHD constants (convenient for setting input parameters)
import Source.PythonMHD_Constants as constants

import Source.PrimCons_HelperModule as primConsLib

#Import numpy for matrix operations
import numpy as np


######INPUT STRUCTURES######

###GRID PARAMETERS###
gridPar = {
            constants.NUM_X_CELLS: 100, #set the number of cells in the x-direction
            constants.MIN_X:-0.5, #set the minimum x-coordinate
            constants.MAX_X:0.5, #set the maximum x-coordinate
            constants.NUM_Y_CELLS: 150, #set the number of cells in the y-direction
            constants.MIN_Y:-0.75, #set the minimum y-coordinate
            constants.MAX_Y:0.75, #set the maximum y-coordinate
            constants.NUM_Z_CELLS: 100, #set the number of cells in the z-direction
            constants.MIN_Z: -0.5, #set the minimum z-coordinate
            constants.MAX_Z: 0.5, #set the maximum z-coordinate
            constants.BC_X:constants.PERIODIC, #set the boundary condition in the x-direction
            constants.BC_Y:constants.PERIODIC, #set the boundary condition in the y-direction
            constants.BC_Z:constants.PERIODIC, #set the boundary condition in the z-direction
          }

###SIMULATION PARAMETERS###
simPar = {
              constants.IS_MHD:False, #boolean flag for whether the simulation has magnetic fields
              constants.GAMMA: 1.66667, #specific heat ratio for the ideal gas
              constants.TLIM: 1.0, #time at which the simulation should terminate
              constants.CFL: 0.4, #CFL number for calculating timestep sizes
              constants.MAX_CYCLES:100000, #max number of simulation cycles
              constants.RECONSTRUCT_ORDER:constants.PPM_SPATIAL_RECONSTRUCTION, #spatial reconstruction order
              constants.ENTROPY_FIX:False, #boolean flag for whether we should apply an entropy fix to small wavespeeds
              constants.EPSILON:0.5, #smallest wavespeed that will not be affected by the entropy fix
              constants.PLOT_DATA:True, #boolean flag for whether we want to visualize the simulation data
              constants.PLOT_DATA_DT:0.05, #amount of simulation time between visualization outputs
              constants.SAVE_FIGS:True, #boolean flag for whether we should save the visualization outputs
              constants.SAVE_FIGS_FORMAT: constants.IMAGE_FORMAT_PNG, #file type for images of any visualization outputs
              constants.SAVE_FIGS_DPI: 300, #resolution in dots-per-inch/dpi for output images
              constants.MATPLOTLIB_BACKEND: constants.MATPLOTLIB_BACKEND_TKAGG, #matplotlib backend (for visualizations)
              constants.SAVE_DATA:True, #boolean flag for whether we should save the numerical data for the simulation
              constants.SAVE_DATA_DT:0.07, #amount of simulation time between numerical data saves
              constants.SAVE_DATA_FORMAT: constants.DATA_FORMAT_MAT, #file type for numerical data saves
              constants.OUTPUT_FOLDER_NAME: "HydroBlast_3D", #name for the outputs folder
              constants.OUTPUT_FILE_NAME: "HydroBlast_3D", #prefix name for the output files
         }

###VISUALIZATION PARAMETERS###
visPar = {
            constants.FIGURES: #array of Figure structures
            [
                #Figure 1
                {
                   constants.FIGURE_TITLE: "3D Hydro Blast", #title for the figure
                   constants.PLOTS: #array of plots to display on this figure
                    [
                       {
                           constants.PLOT_VAR:constants.DENSITY #gas variable we want to plot (density)
                       },
                       {
                           constants.PLOT_VAR:constants.PRESSURE #gas variable we want to plot (pressure)
                       },
                       {
                            constants.PLOT_VAR:constants.DENSITY, #gas variable we want to plot (density)
                            constants.PLOT_TYPE: constants.PLOT_TYPE_2D, #type of plot (2D colormap)
                            constants.PLOT_PLANE: constants.XY_PLANE, #plane that is parallel with our
                                                                      #2D cross-section (xy-plane)
                            constants.Z_POS: 0, #z-position of the cross-section
                       },
                       {
                            constants.PLOT_VAR:constants.PRESSURE, #gas variable we want to plot (pressure)
                            constants.PLOT_TYPE: constants.PLOT_TYPE_1D, #type of plot (1D line plot)
                            constants.PLOT_AXIS: constants.Z_AXIS, #axis that is parallel with our line (z-axis)
                            constants.X_POS: 0, #x-position of the line
                            constants.Y_POS: 0, #y-position of the line
                       }
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
#Purpose: Creates the primitive variable matrix for the 3D hydro blast,
#         using the parameters in gridPar and blastPar.
#Input Parameters: gridPar (the grid parameters structure)
#                  blastPar (the 3D Hydro Blast parameters structure)
#Outputs: primVars (the primitive variable matrix that has been
#                   constructed based on the specifications in
#                   gridPar and blastPar)
def getBlastPrimVars(gridPar, blastPar):
    #Get the number of cells in the x-direction
    numXCells = gridPar["numXCells"]
    assert(numXCells > 0)
    #Get the number of cells in the y-direction
    numYCells = gridPar["numYCells"]
    assert (numYCells > 0)
    #Get the number of cells in the z-direction
    numZCells = gridPar["numZCells"]
    assert(numZCells > 0)
    #Get the x-, y-, and z-coordinates for every cell in the simulation grid
    (xCoords, yCoords, zCoords) = initHelper.createCoordinates3D_Cartesian(gridPar)
    #Calculate each cell's distance from the centre of the simulation grid (at coordinates (x = 0, y = 0, z = 0)
    radii = np.sqrt(xCoords*xCoords + yCoords*yCoords + zCoords*zCoords)

    #Get the blast parameteres
    blastRadius = blastPar["blastRadius"]
    ambPres = blastPar["ambPres"]
    presRatio = blastPar["presRatio"]
    ambDens = blastPar["ambDens"]
    densRatio = blastPar["densRatio"]

    #Create the density values
    rho = ambDens*np.ones(shape=(numYCells,numXCells,numZCells))
    rho[radii < blastRadius] = densRatio*ambDens
    #Set the x-velocities to zero
    vx = np.zeros(shape=(numYCells,numXCells,numZCells))
    #Set the y-velocities to zero
    vy = np.zeros(shape=(numYCells,numXCells,numZCells))
    #Set the z-velocities to zero
    vz = np.zeros(shape=(numYCells,numXCells,numZCells))
    #Create the pressure values
    pres = ambPres*np.ones(shape=(numYCells,numXCells,numZCells))
    pres[radii < blastRadius] = presRatio*ambPres
    #Create the primitive variables matrix,
    #which will contain the five hydrodynamic
    #primitive variables in the following
    #order: density, x-velocity, y-velocity,
    #       z-velocity, and pressure
    primVars = np.zeros(shape=(5,numYCells,numXCells,numZCells))
    primVars[0,:] = rho
    primVars[1,:] = vx
    primVars[2,:] = vy
    primVars[3,:] = vz
    primVars[4,:] = pres
    #Return the primitive variables matrix
    return primVars

######MAIN SCRIPT######

#Build the primitive variables matrix for
#the initial state of the 3D Hydro Blast
blastPrimVars = getBlastPrimVars(gridPar, blastPar)

#Create the Simulation object
blastSim = Simulation(simPar,gridPar,blastPrimVars,visPar)

#Run the simulation
newGrid = blastSim.run()


