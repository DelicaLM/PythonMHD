#Simulation.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
# Purpose: Defines the main Simulation class in PythonMHD. When a user wants to run a PythonMHD simulation, they first
#          need to create an instance of the Simulation class (see PythonMHD user guide for how to create the Simulation
#          object in your initialization file). After creating this object, the user just needs to call the public
#          "run" method that is defined in this class.
# Current Functionality: PythonMHD currently supports the following simulation types:
#                        -1D Hydro simulations (with or without visualizations)
#                        -1D MHD simulations (with or without visualizations)
#                        -2D Hydro simulations (with or without visualizations) (with or without parallelization)
#                        -2D MHD simulations (with or without visualizations) (with or without parallelization)
#                        -3D Hydro simulations (with or without visualizations) (with or without parallelization)
#                        -3D MHD simulations (with or without visualizations) (with or without parallelization)
#
# References: Many publications guided the development of PythonMHD. The numerical algorithms in this
#             version of PythonMHD are intended to duplicate the results that one would obtain with the
#             2017 version of Athena by Stone et al. [1]. The source code on the Athena GitHub
#             page [1] and the 2008 Athena methods paper [2] are the primary resources that were used to ensure that
#             PythonMHD and the 2017 C-version of Athena generate the same outputs for standard hydrodynamic
#             and MHD test problems. The other resources that were consulted in the creation of PythonMHD are
#             referenced in the various helper modules (e.g., RiemannSolvers_MHD_Roe_HelperModule.py) that implement
#             the numerical algorithms for hydrodynamic and MHD simulations).
#             1. https://github.com/PrincetonUniversity/Athena-Cversion
#             2. Stone, J. M., Gardiner, T. A., Teuben, P., Hawley, J. F., & Simon, J. B. (2008).
#                Athena: A new code for astrophysical MHD. The Astrophysical Journal Supplemental Series,
#                178(1), 137-177. https://iopscience.iop.org/article/10.1086/588755/pdf.
#
# Acknowledgements:
#

##########IMPORT STATEMENTS###########

#Import numpy for matrix operations
import numpy as np

#Import time for keeping track of how long simulations are taking
import time as timeLib

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants

#Import SimulationGrid class (for creating a SimulationGrid object that stores all the data for the Simulation)
from Source.SimulationGrid import SimulationGrid

#Import the Simulation class helper module (for validating the user's simulation parameters)
import Source.Simulation_HelperModule as simLib

#Import the SimVisualizer class for visualizing simulations
from Source.SimVisualizer import SimVisualizer

#Import the FileSaver class for saving visualizations and numerical data
from Source.FileSaver import FileSaver

#Import the evolve functions (i.e., functions that advance/evolve the simulation through time)
#for hydrodynamic simulations
import Source.EvolveFunctions_Hydro as evolveHydro

#Import the evolve functions for MHD simulations
import Source.EvolveFunctions_MHD as evolveMHD

#############SIMULATION CLASS##############
#Class: Simulation
#Purpose: A Simulation object provides all of the necessary functions for running a PythonMHD simulation
#         of any type. Upon creating a new Simulation, this class will automatically create a new instance
#         of the SimulationGrid class.
class Simulation:
    #########CONSTRUCTOR METHOD##########
    # Function: __init__
    # Purpose: Creates a new Simulation object (if the user provides valid inputs for
    #          simPar, gridPar, and the primitive variables matrix).
    # Inputs: simPar (the dictionary that contains the user's simulation parameters
    #                (see PythonMHD user guide for how to define simPar in your
    #                 initialization file)) (REQUIRED)
    #         gridPar (the dictionary that contains the user's grid parameters
    #                 (see PythonMHD user guide for how to define simPar in your
    #                  initialization file)) (REQUIRED)
    #         primVars (the primitive variables for every cell in the simulation grid
    #                   (see PythonMHD user guide for how to set your initial conditions
    #                    with the primVars) (REQUIRED)
    #         visPar (the visualization parameters) (OPTIONAL) (ONLY REQUIRED IF plotData
    #                                                           IS TRUE IN simPar)
    #         faceBx (the face-centred Bx magnetic field values) (OPTIONAL)
    #                (If you do not provide a faceBx matrix, PythonMHD will calculate the face-centred field values
    #                 for you with the Constrained Transport (CT) method (see CT_HelperModule.py for additional details).
    #                 Only set your own initial face-centred magnetic field values if you are concerned about the tiny
    #                 numerical differences that result from calculating the initial face-centred values with the CT
    #                 method's averaging formula. These numerical differences will only be relevant if you want to compare
    #                 PythonMHD's outputs against another MHD code with an extremely high level of precision (~10^-16).
    #                 I personally had to directly set the faceBx and faceBy values when I was comparing PythonMHD
    #                 against Athena for the Orszag-Tang vortex problem, because tiny differences in the initial faceBx
    #                 and faceBy conditions made it a lot harder to figure out whether there were real/significant
    #                 differences between my code and Athena.)
    #         faceBy (the face-centred By magnetic field values) (OPTIONAL)
    #                (If you do not provide a faceBy matrix, PythonMHD will calculate the face-centred field values
    #                 for you with the Constrained Transport (CT) method (see CT_HelperModule.py for additional details).
    #         faceBz (the face-centred Bz magnetic field values) (OPTIONAL)
    #                (If you do not provide a faceBz matrix, PythonMHD will calculate the face-centred field values
    #                 for you with the Constrained Transport (CT) method (see CT_HelperModule.py for additional details).
    def __init__(self, simPar, gridPar, primVars, visPar={},
                 consVars=np.zeros(shape=(1)),
                 faceBx=np.zeros(shape=(1)),faceBy=np.zeros(shape=(1)),faceBz=np.zeros(shape=(1))):
        print("Welcome to PythonMHD (version 2024.1)\n")
        #Use the Simulation helper module to validate the user's simulation parameters
        simLib.validateSimPar(simPar, primVars)
        #Check if the simulation is MHD
        self.isMHD = constants.DEFAULT_IS_MHD
        #If the user has not set isMHD in their simPar structure,
        #we assume that the simulation does not have magnetic fields.
        if constants.IS_MHD in simPar.keys():
            self.isMHD = simPar[constants.IS_MHD]
        #Get the time at which the simulation should end
        #Assuming nothing goes wrong (e.g., no problematic numerical instabilities, we don't reach the max number
        #of simulation cycles, etc.), PythonMHD will evolve the simulation from time t = 0 to t = tLim.
        self.tLim = simPar[constants.TLIM]
        #Get the max number of cycles if the user wants to override the default value
        self.maxCycles = constants.DEFAULT_MAX_CYCLES
        if constants.MAX_CYCLES in simPar.keys():
            self.maxCycles = simPar[constants.MAX_CYCLES]
        #Get the spatial reconstruction order for the simulation
        self.reconstructOrder = constants.NO_SPATIAL_RECONSTRUCTION
        if constants.RECONSTRUCT_ORDER in simPar.keys():
            self.reconstructOrder = simPar[constants.RECONSTRUCT_ORDER]
        #Get the Riemann solver that should be used for the simulation
        #(Currently, PythonMHD only supports Roe's Riemann solver. Future versions
        # of the code will provide additional Riemann solver options.)
        # self.riemannSolver = constants.ROE
        # if constants.RIEMANN_SOLVER in simPar.keys():
        #     self.riemannSolver = simPar[constants.RIEMANN_SOLVER]
        #Get the specific heat ratio for the gas
        if constants.GAMMA in simPar.keys():
            self.gamma = simPar[constants.GAMMA]
        #Check whether the user wants to apply an entropy fix to small wavespeeds in order to minimize the
        #likelihood of unphysical rarefaction shocks by increasing artificial viscosity (see
        #RiemannSolvers_Hydro_Roe_HelperModule.py for information on when you might want to use and entropy fix)
        self.entropyFix = False
        if constants.ENTROPY_FIX in simPar.keys():
            self.entropyFix = simPar[constants.ENTROPY_FIX]
        #Check whether the user has provided their own epsilon value for the entropy fix
        self.epsilon = constants.DEFAULT_EPSILON
        if constants.EPSILON in simPar.keys():
            self.epsilon = simPar[constants.EPSILON]
        #Check whether the user has set a minimum density value for the simulation
        #(which will override PythonMHD's default minimum density value)
        self.minDens = constants.DEFAULT_MIN_DENSITY
        if constants.MIN_DENSITY in simPar.keys():
            self.minDens = simPar[constants.MIN_DENSITY]
        #Check whether the user has set a minimum pressure value for the simulation
        #(which will override PythonMHD's default minimum pressure value)
        self.minPres = constants.DEFAULT_MIN_PRESSURE
        if constants.MIN_PRESSURE in simPar.keys():
            self.minPres = simPar[constants.MIN_PRESSURE]
        #Check whether the user has set a minimum energy value for the simulation
        #(which will override PythonMHD's default minimum energy value)
        self.minEnergy = constants.DEFAULT_MIN_ENERGY
        if constants.MIN_ENERGY in simPar.keys():
            self.minEnergy = simPar[constants.MIN_ENERGY]
        #Create the SimulationGrid object for the simulation
        print("Building Simulation Grid\n")
        self.grid = SimulationGrid(primVars,gridPar,self.isMHD,self.gamma,self.minDens,self.minPres,self.minEnergy,
                                   consVars,faceBx,faceBy,faceBz)
        #The default CFL number for the simulation depends on whether it is 1D, 2D, or 3D
        if self.grid.nDim == 1:
            self.cfl = constants.DEFAULT_CFL_1D
        elif self.grid.nDim == 2:
            self.cfl = constants.DEFAULT_CFL_2D
        else:
            self.cfl = constants.DEFAULT_CFL_3D
        #Check if the user has set their own CFL number
        if constants.CFL in simPar.keys():
            self.cfl = simPar[constants.CFL]
        #Check if the user wants to plot data during their simulation
        self.plotData = constants.DEFAULT_PLOT_DATA
        if constants.PLOT_DATA in simPar.keys():
            self.plotData = simPar[constants.PLOT_DATA]
        #Set the makeMovie flag to false by default
        self.makeMovie = False
        #Set the save figures flag to False (in case we are not plotting data)
        self.saveFigs = False
        if self.plotData:
            self.saveFigs = constants.DEFAULT_SAVE_FIGS
            if constants.SAVE_FIGS in simPar.keys():
                self.saveFigs = simPar[constants.SAVE_FIGS]
        #Check if the user wants a particular file type (e.g., ".png", ".pdf", etc.)
        self.saveFigsFormat = constants.DEFAULT_IMAGE_FORMAT
        if constants.SAVE_FIGS_FORMAT in simPar.keys():
            self.saveFigsFormat = simPar[constants.SAVE_FIGS_FORMAT].replace(".","")
        #Check if the user wants a particular resolution (in dpi/dots-per-inch)
        self.saveFigsDpi = constants.DEFAULT_SAVE_FIGS_DPI
        if constants.SAVE_FIGS_DPI in simPar.keys():
            self.saveFigsDpi = simPar[constants.SAVE_FIGS_DPI]
        #If the user wants to plot data, we need to create a SimVisualizer
        #object to manage the visualizations.
        if self.plotData:
            #Check how frequently the user wants to plot their simulation data
            #(e.g., every 0.1 simulation time units, every 0.5, etc.)
            self.plotDt = simPar[constants.PLOT_DATA_DT]
            #Check if the user wants to override the default matplotlib backend
            self.matplotlibBackend = constants.DEFAULT_MATPLOTLIB_BACKEND
            if constants.MATPLOTLIB_BACKEND in simPar.keys():
                self.matplotlibBackend = simPar[constants.MATPLOTLIB_BACKEND]
            #Check if the user wants to make a movie with any saved figures
            if self.saveFigs:
                self.makeMovie = constants.DEFAULT_MAKE_MOVIE
                if constants.MAKE_MOVIE in simPar.keys():
                    self.makeMovie = simPar[constants.MAKE_MOVIE]
            #Create the SimVisualizer object
            self.Visualizer = SimVisualizer(self.grid, visPar, self.plotDt, self.tLim,
                                            self.matplotlibBackend, self.saveFigs, self.saveFigsFormat,
                                            self.saveFigsDpi, self.makeMovie)
            #Check if the user wants to make a movie with any saved figures
            if self.saveFigs:
                self.makeMovie = constants.DEFAULT_MAKE_MOVIE
                if constants.MAKE_MOVIE in simPar.keys():
                    self.makeMovie = simPar[constants.MAKE_MOVIE]
        #Check if the user wants to save all of the numerical data for the
        #simulation in a .mat file (additional file types will be provided
        #in future versions of PythonMHD).
        self.saveData = False
        if constants.SAVE_DATA in simPar.keys():
            self.saveData = simPar[constants.SAVE_DATA]
        #If the user wants to save their numerical outputs, we need to
        #know how frequently we should save the simulation data
        #(e.g., every 0.1 simulation time units, every 0.5, etc.)
        if self.saveData:
            if constants.SAVE_DATA_DT in simPar.keys():
                self.saveDt = simPar[constants.SAVE_DATA_DT]
            else:
                #If the user has not provided a saveDt value,
                #we will use the visualization timestep plotDt.
                #(Note: The simPar validation function ensures that plotDt is defined
                #       in this scenario.)
                self.saveDt = self.plotDt
            self.saveDataFormat = constants.DEFAULT_DATA_FORMAT
            if constants.SAVE_DATA_FORMAT in simPar.keys():
                self.saveDataFormat = simPar[constants.SAVE_DATA_FORMAT]
        #If the user wants to save visualizations or numerical data, we need to create
        #a FileSaver object to deal with directory management and other file creation details
        if (self.plotData and self.saveFigs) or self.saveData:
            #Check if the user has passed parameters in simPar for the path to where the outputs folder should be
            #be created, the name that the outputs folder should have, and the prefix name for all files that are
            #saved in the outputs folder
            outputFolderPath = ""
            outputFolderName = ""
            outputFileName = ""
            if constants.OUTPUT_FOLDER_PATH in simPar.keys():
                outputFolderPath = simPar[constants.OUTPUT_FOLDER_PATH]
            if constants.OUTPUT_FOLDER_NAME in simPar.keys():
                outputFolderName = simPar[constants.OUTPUT_FOLDER_NAME]
            if constants.OUTPUT_FILE_NAME in simPar.keys():
                outputFileName = simPar[constants.OUTPUT_FILE_NAME]
            #We need to tell the file saver how frequently it will be saving images and/or data
            saveFigDt = -1.0 #set the save figure timestep to -1 if we are not saving figures
            saveDataDt = -1.0 #set the save data timestep to -1 if we are not saving numerical data
            #If we are saving visualizations, the time between visualization saves should be the same
            #as the time between plots.
            if self.plotData:
                saveFigDt = self.plotDt
            #Get the saveDt time if we are saving numerical data
            if self.saveData:
                saveDataDt = self.saveDt
            #Create the file saver object
            self.FileSaver = FileSaver(self.saveFigs, self.saveData, self.tLim, saveFigDt, saveDataDt,
                                       outputFolderPath, outputFolderName, outputFileName)


    #########PUBLIC RUN METHOD##########
    #Function: run
    #Input Parameters: None (all necessary details are stored in the object
    #                        that calls the run function)
    #Outputs: In addition to any visualizations or saved data files, this
    #         function returns a SimulationGrid object that contains
    #         all of the primitive and conservative variables at the end of the
    #         simulation (i.e., the final, fully-evolved state,
    #         assuming no errors occurred that caused the simulation
    #         to end before reaching the requested time limit).
    def run(self):
        #Call the private run_sim function
        return self.__run_sim()

    #########PRIVATE RUN METHOD##########
    #Function: run_sim
    #Input Parameters: None (all necessary details are stored in the object
    #                        that calls the run function)
    #Outputs: In addition to any visualizations or saved data files, this
    #         function returns a SimulationGrid object that contains
    #         all of the primitive and conservative variables at the end of the
    #         simulation (i.e., the final, fully-evolved state,
    #         assuming no errors occurred that caused the simulation
    #         to end before reaching the requested time limit).
    def __run_sim(self):
        print("Starting simulation\n")
        #Keep track of how long the simulation takes
        startTime = timeLib.time()
        #Set the initial simulation time and cycle to zero
        simTime = 0
        numCycles = 0
        #Retrieve the time limit for the simulation
        simTimeLim = self.tLim
        #If we are plotting simulation data, show the user the
        #initial state of the system.
        if self.plotData:
            self.Visualizer.visualize(self.grid, simTime)
            #Save the initial state images if the user wants to save their visualizations
            if self.saveFigs:
                self.FileSaver.saveFigures(self.Visualizer, simTime)
        #If we are saving numerical data, save the initial state of the gas system.
        if self.saveData:
            self.FileSaver.saveDataFiles(self.grid, simTime, self.saveDataFormat)
        #Figure out which evolve function we should use for the simulation
        #(1D hydro, 2D hydro, 3D hydro, 1D MHD, 2D MHD, or 3D MHD)
        evolveFunction = evolveHydro.evolve_1D_hydro
        if not self.isMHD:
            if self.grid.nDim == 2:
                evolveFunction = evolveHydro.evolve_2D_hydro
            elif self.grid.nDim == 3:
                evolveFunction = evolveHydro.evolve_3D_hydro
        else:
            if self.grid.nDim == 1:
                evolveFunction = evolveMHD.evolve_1D_mhd
            if self.grid.nDim == 2:
                evolveFunction = evolveMHD.evolve_2D_mhd
            elif self.grid.nDim == 3:
                evolveFunction = evolveMHD.evolve_3D_mhd
        #Figure out when we are next supposed to plot and/or save data
        nextPlotTime = self.tLim
        lastPlotTime = 0.0
        nextSaveTime = self.tLim
        lastSaveTime = 0.0
        plotTimePrec = -1
        saveTimePrec = -1
        if self.plotData:
            nextPlotTime = np.minimum(nextPlotTime,self.plotDt)
            plotTimePrec = self.Visualizer.timePrec
        if self.saveData:
            nextSaveTime = np.minimum(nextSaveTime,self.saveDt)
            saveTimePrec = self.FileSaver.dataTimePrec
        #Figure out when we need to stop the simulation to plot or save data
        nextStopTime = np.minimum(nextPlotTime,nextSaveTime)
        #Initialize the new primitive and conservative variables that will
        #be returned to the user at the end of the simulation
        newPrimVars = self.grid.primVars
        newConsVars = self.grid.consVars
        #Keep evolving the gas until we reach the specified time limit or hit the max number of simulation cycle
        while simTime < simTimeLim and numCycles < self.maxCycles:
            #Call the evolve function (the number of outputs depends on the number of face-centred magnetic field
            #components that need to be returned)
            if not self.isMHD or self.grid.nDim == 1:
                (newPrimVars, newConsVars, newSimTime, newCycle,
                 secondLastPrimVars, secondLastConsVars, secondLastTime) = evolveFunction(self.grid, simTime,
                                                                                          nextStopTime, numCycles,
                                                                                          self.maxCycles,
                                                                                          self.gamma, self.cfl,
                                                                                          self.minDens, self.minPres,
                                                                                          self.minEnergy,
                                                                                          self.reconstructOrder,
                                                                                          self.entropyFix,
                                                                                          self.epsilon)
            elif self.grid.nDim == 2:
                (newPrimVars, newConsVars, newFaceBx, newFaceBy, newSimTime, newCycle,
                 secondLastPrimVars, secondLastConsVars, secondLastFaceBx,
                 secondLastFaceBy, secondLastTime) = evolveFunction(self.grid, simTime, nextStopTime, numCycles,
                                                                    self.maxCycles, self.gamma, self.cfl, self.minDens,
                                                                    self.minPres, self.minEnergy, self.reconstructOrder,
                                                                    self.entropyFix, self.epsilon)
                self.grid.faceBx = newFaceBx
                self.grid.faceBy = newFaceBy
            else: #if the simulation is 3D and MHD
                (newPrimVars, newConsVars, newFaceBx, newFaceBy, newFaceBz, newSimTime, newCycle,
                 secondLastPrimVars, secondLastConsVars, secondLastFaceBx, secondLastFaceBy,
                 secondLastFaceBz, secondLastTime) = evolveFunction(self.grid, simTime, nextStopTime, numCycles,
                                                                    self.maxCycles, self.gamma, self.cfl, self.minDens,
                                                                    self.minPres, self.minEnergy, self.reconstructOrder,
                                                                    self.entropyFix, self.epsilon)
                self.grid.faceBx = newFaceBx
                self.grid.faceBy = newFaceBy
                self.grid.faceBz = newFaceBz
            #Update the SimulationGrid object with the new primitive and conservative variables
            self.grid.primVars = newPrimVars
            self.grid.consVars = newConsVars
            #Update the simulation time and cycle number
            simTime = newSimTime
            numCycles = newCycle
            #Check if we should save data
            if self.saveData and nextStopTime == nextSaveTime:
                lastSaveTime = simTime
                self.FileSaver.saveDataFiles(self.grid, simTime, self.saveDataFormat)
            #Check if we should plot data
            if self.plotData and nextStopTime == nextPlotTime:
                lastPlotTime = simTime
                self.Visualizer.visualize(self.grid, simTime)
                if self.saveFigs:
                    self.FileSaver.saveFigures(self.Visualizer,simTime)
            #Calculate the next saving and plotting times
            nextPlotTime = self.tLim
            nextSaveTime = self.tLim
            if self.plotData:
                nextPlotTime = np.round(np.minimum(nextPlotTime, lastPlotTime + self.plotDt), plotTimePrec)
            if self.saveData:
                nextSaveTime = np.round(np.minimum(nextSaveTime, lastSaveTime + self.saveDt), saveTimePrec)
            nextStopTime = np.minimum(nextPlotTime, nextSaveTime)
            #If we are still simulating, use the gas variables from the last cycle before the visualization/data saving
            #step (because the small timesteps used to reach the exact plotting/saving time can have a negative long-term
            #impact on the simulations if we don't revert back to the second-last cycle).
            if simTime < simTimeLim and numCycles < self.maxCycles:
                self.grid.primVars = secondLastPrimVars
                self.grid.consVars = secondLastConsVars
                simTime = secondLastTime
                if self.isMHD and self.grid.nDim >= 2:
                    self.grid.faceBx = secondLastFaceBx
                    self.grid.faceBy = secondLastFaceBy
                    if self.grid.nDim == 3:
                        self.grid.faceBz = secondLastFaceBz
        #Update the SimulationGrid with the final gas state
        self.grid.primVars = newPrimVars
        self.grid.consVars = newConsVars
        if self.isMHD and self.grid.nDim >= 2:
            self.grid.faceBx = newFaceBx
            self.grid.faceBy = newFaceBy
            if self.grid.nDim == 3:
                self.grid.faceBz = newFaceBz
        #Tell the user how long the simulation took
        endTime = timeLib.time()
        print("Simulation took: ", endTime - startTime, " seconds")
        #Check if we need to make a movie of any visualization outputs
        if self.plotData and self.saveFigs and self.makeMovie:
            print("MAKING MOVIES (DO NOT END SIM YET)")
            self.FileSaver.makeMovies(self.Visualizer)
        print("SIMULATION PROCESSING COMPLETE. SAFE TO END PROGRAM.")
        #Keep displaying output visualizations until the user closes
        #the matplotlib window(s)
        if self.plotData:
            self.Visualizer.blockOnPlot()
        return self.grid






