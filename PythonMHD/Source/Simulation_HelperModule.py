#Simulation_HelperModule.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides supporting functions for validating the user's simulation parameters (simPar) in the Simulation
#         class that is defined in Simulation.py.

######IMPORT STATEMENTS######

#Import Numpy for matrix operations
import numpy as np

#Import os for checking if output folder paths actually exist
import os

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants

######SIMULATION PARAMETERS VALIDATION FUNCTION#######
#Function: validateSIMPAR
#Purpose: Checks whether the user has submitted valid inputs for simPar (when compared
#         against their primitive variable matrix). If there are any problems (e.g.,
#         the CFL number is invalid for the number of spatial dimensions), a
#         RuntimeError will be raised.
#Input Parameters: simPar (the simulation parameters dictionary to check)
#                  primVars (the primitive variables matrix)
#Outputs: void (this function will raise a RuntimeError if simPar
#               fails the validation test)
def validateSimPar(simPar,primVars):
    print("Validating Simulation Parameters\n")
    #If the user has provided an isMHD parameter, make sure it is a boolean.
    if constants.IS_MHD in simPar.keys():
        if type(simPar[constants.IS_MHD]) != bool:
            raise RuntimeError("\nYou passed an invalid value for " + str(constants.IS_MHD) + " in your simulation parameters "
                               + "structure. The only accepted values for isMHD are True or False. \nSet isMHD to True if you "
                               + "want to include magnetic fields in your simulation. The default value of isMHD "
                               + "is False, which means that \nPythonMHD will assume that your simulation is hydrodynamic "
                               + "if you remove the isMHD parameter from simPar. You can make isMHD = True \nthe default "
                               + "setting if you change DEFAULT_IS_MHD from False to True in PythonMHD_Constants.py.")
    #Raise an error if the user has not specified the time limit at which the simulation should terminate
    if constants.TLIM not in simPar.keys():
        raise RuntimeError("You did not specify a time limit (" + constants.TLIM + ") in your simulation parameters "
                           + "structure. You must provide an end time for your simulation in simPar (e.g., tLim = 1.0).")
    else: #Make sure the time limit is a numeric value greater than 0
        if type(simPar[constants.TLIM]) != int and type(simPar[constants.TLIM]) != float:
            raise RuntimeError("\nYou did not provide a valid data type for " + constants.TLIM + " in your simulation "
                               + "parameters structure.\nYou must pass a numeric value for tLim that is greater than 0.")
        else:
            if simPar[constants.TLIM] <= 0.0:
                raise RuntimeError("\nYou passed a value for " + constants.TLIM + " in your simulation parameters structure "
                                   + "that is too small (<= 0).\nThe time limit for any simulation must be greater than 0.\n")
    #Check if the user has set a max number of simulation cycles
    if constants.MAX_CYCLES in simPar.keys():
        #Make sure the max number of cycles is an int greater than zero
        if type(simPar[constants.MAX_CYCLES]) != int:
            raise RuntimeError("\nYou passed an invalid data type for " + constants.MAX_CYCLES + " in your simulation "
                               + "parameters structure.\nYou can only pass integer values for the max number of simulation "
                               + "cycles.")
        else:
            if simPar[constants.MAX_CYCLES] < 1:
                raise RuntimeError("\nYou passed an invalid value for " + constants.MAX_CYCLES + " in your simulation "
                                   + "parameters structure.\nThe max number of simulation cycles must be greater than 0.\n")
    #Make sure the primitive variables matrix is a numpy array
    if not isinstance(primVars, np.ndarray):
        raise RuntimeError("\nYou did not provide a valid primitive variable matrix.\nPythonMHD only accepts "
                           + "primitive variables that are submitted as a numpy matrix/ndarray.\nSee sample "
                           + "initialization files and the PythonMHD user guide for how to format your primitive variables.")
    else: #if the primitive variables are a matrix/array object
        #Get the number of spatial dimensions in the primitive variables matrix
        nDim = len(np.squeeze(primVars).shape) - 1
        #Raise an error if the simulation isn't 1D, 2D, or 3D
        if nDim != 1 and nDim != 2 and nDim != 3:
            raise RuntimeError("\nYou have an invalid number of spatial dimensions in your primitive variables matrix. "
                               + "\nPythonMHD only supports 1D, 2D, or 3D simulations.")
        else:
            #Check if the user has set their own CFL number
            if constants.CFL in simPar.keys():
                #Make sure that the CFL number is a float
                if type(simPar[constants.CFL]) != float:
                    raise RuntimeError("\nYou passed an invalid data type for " + constants.CFL + " in your simulation parameters structure. "
                                       + "\nThe CFL number for your simulation must be a floating point value > 0 and < 1 "
                                       + "(> 0 and < 0.5 for 3D MHD simulations).")
                else:
                    #Make sure the CFL number is greater than zero
                    if simPar[constants.CFL] <= 0:
                        raise RuntimeError("\nYou passed an invalid value for " + constants.CFL + " in your simulation parameters structure. "
                                           + "\nThe CFL number for the simulation must be greater than 0.")
                    else:
                        #If the simulation is 3D, the CFl number must be less than 0.5.
                        if nDim == 3:
                            if simPar[constants.CFL] >= 0.5:
                                raise RuntimeError( "\nYou passed an invalid value for " + constants.CFL + " in your simulation parameters "
                                                     + "structure.\nThe CFL number for a 3D simulation must be less "
                                                     + "than 0.5.")
                        else:
                            #For any type of simulation, the CFL number must be less than 1
                            if simPar[constants.CFL] >= 1.0:
                                raise RuntimeError("\nYou passed an invalid value for " + constants.CFL + " in your simulation parameters "
                                                     + "structure.\nThe CFL number for your simulation must be less "
                                                     + "than 1.")
    #Check if the user has provided their own minimum density value
    if constants.MIN_DENSITY in simPar.keys():
        #Make sure the minimum density is a floating point number
        if type(simPar[constants.MIN_DENSITY]) != float:
            raise RuntimeError("You passed an invalid data type for " + constants.MIN_DENSITY + " in your simulation parameters structure."
                               + " The minimum density value for your simulation must be a small floating point value"
                               + " > 0. We recommend choosing a value less than 10^-5. The default minimum density"
                               + " in PythonMHD is " + str(constants.DEFAULT_MIN_DENSITY) + ". We need to use a small value"
                               + " because PythonMHD will modify any density values that fall below this minimum"
                                 " (i.e., dens = {dens, dens >= minDens; minDens, dens < minDens}). You should only"
                               + " use this parameter as a final guard against numerical instabilities that generate"
                               + " negative densities. PythonMHD will tell you whenever (and wherever) it had to"
                               + " apply the minimum density value. If the density threshold is being applied "
                               + " frequently, you might want to reconsider your initial conditions (see PythonMHD's "
                               + " user guide for information on how to avoid setting initial conditions that increase"
                               + " the likelihood of unphysical numerical instabilities). You also should reconsider"
                               + " your initial conditions if you need to override PythonMHD's minimum density value,"
                               + " because the built-in value was chosen to address numerical errors on the order"
                               + " of magnitude that we can reasonably expect for a well-functioning simulation."
                               + " If you need to increase the default value to prevent negative densities, there"
                               + " is probably a deeper issue with your simulation (see PythonMHD's user guide for"
                               + " details on how you can resolve the most common issues that are known to generate"
                               + " unphysical density values).")
        else:
            #Make sure that the minimum density value is greater than zero
            if simPar[constants.MIN_DENSITY] <= 0:
                raise RuntimeError("You passed an invalid value for " + constants.MIN_DENSITY + " in your simulation parameters structure."
                                + " The minimum density value for your simulation must be a small floating point value"
                                + " > 0. We recommend choosing a value less than 10^-5. The default minimum density"
                                + " in PythonMHD is " + str(constants.DEFAULT_MIN_DENSITY) + ". We need to use a small value"
                                + " because PythonMHD will modify any density values that fall below this minimum"
                                + " (i.e., dens = {dens, dens >= minDens; minDens, dens < minDens}). You should only"
                                + " use this parameter as a final guard against numerical instabilities that generate"
                                + " negative densities. PythonMHD will tell you whenever (and wherever) it had to"
                                + " apply the minimum density value. If the density threshold is being applied "
                                + " frequently, you might want to reconsider your initial conditions (see PythonMHD's "
                                + " user guide for information on how to avoid setting initial conditions that increase"
                                + " the likelihood of unphysical numerical instabilities). You also should reconsider"
                                + " your initial conditions if you need to override PythonMHD's minimum density value,"
                                + " because the built-in value was chosen to address numerical errors on the order"
                                + " of magnitude that we can reasonably expect for a well-functioning simulation."
                                + " If you need to increase the default value to prevent negative densities, there"
                                + " is probably a deeper issue with your simulation (see PythonMHD's user guide for"
                                + " details on how you can resolve the most common issues that are known to generate"
                                + " unphysical density values).")
            elif simPar[constants.MIN_DENSITY] < constants.DEFAULT_MIN_DENSITY:
                #Warn the user if their minimum density value is less than the default minimum density
                print("Runtime Warning: You passed a value for " + constants.MIN_DENSITY + " in your simulation parameters structure that is"
                                   + " less than PythonMHD's default minimum density value. PythonMHD presents this"
                                   + " warning because the default minimum density value was chosen to address numerical"
                                   + " errors on the order of magnitude that we can reasonably expect for a"
                                   + " well-functioning simulation. The density values in your simulation are getting"
                                   + " smaller than they should be if you need to set a lower minimum density value."
                                   + " You can most likely fix the issue by adjusting your physical -> simulation units"
                                   + " scaling factors (see PythonMHD's user guide for more information on how to "
                                   + " set initial conditions and unit scaling factors that minimize the likelihood of"
                                   + " unphysical numerical instabilities in your simulations).")
            elif simPar[constants.MIN_DENSITY] > constants.DEFAULT_MIN_DENSITY:
                #Warn the user if their minimum density is greater than the default minimum density
                print("Runtime Warning: You passed a value for " + constants.MIN_DENSITY + " in your simulation parameters structure that is\n"
                    + "                 larger than PythonMHD's default minimum density value. PythonMHD presents this\n"
                    + "                 warning because the default minimum density value was chosen to address numerical\n"
                    + "                 errors on the order of magnitude that we can reasonably expect for a well-functioning\n"
                    + "                 simulation. If you need to increase the default value to prevent negative densities\n"
                    + "                 there could be a deeper issue with your simulation (see PythonMHD's user guide for\n"
                    + "                 details on how you can resolve the most common issues that are known to generate\n"
                    + "                 unphysical density values).\n")
    #Check if the user has specified the spatial reconstruction order
    if constants.RECONSTRUCT_ORDER in simPar.keys():
        #Make sure they have passed an integer for the reconstruction order code
        if type(simPar[constants.RECONSTRUCT_ORDER]) != int:
            raise RuntimeError("\nYou passed an invalid data type for " + constants.RECONSTRUCT_ORDER + " in your simulation parameters "
                               + "structure.\nYou can only pass integer values (no reconstruction = " + str(constants.NO_SPATIAL_RECONSTRUCTION)
                               + ", second-order/PPM reconstruction = " + str(constants.PPM_SPATIAL_RECONSTRUCTION)
                               + ") for the spatial reconstruction order.\nFirst-order/PLM reconstruction will be available "
                               + "in a future version of PythonMHD")
        else:
            #Make sure the integer matches one of the available reconstruction orders
            if simPar[constants.RECONSTRUCT_ORDER] != constants.NO_SPATIAL_RECONSTRUCTION \
            and simPar[constants.RECONSTRUCT_ORDER] != constants.PPM_SPATIAL_RECONSTRUCTION:
                raise RuntimeError("\nYou passed an invalid value for " + constants.RECONSTRUCT_ORDER + " in your simulation parameters "
                                   + "structure.\nThe only accepted values for reconstructOrder are "
                                   + "no reconstruction = " + str(constants.NO_SPATIAL_RECONSTRUCTION) + " and second-order/PPM "
                                   + "reconstruction = "+ str(constants.PPM_SPATIAL_RECONSTRUCTION) + ".\nFirst-order/PLM "
                                   + "reconstruction will be available in a future version of PythonMHD.")
    #If the user has requested a particular Riemann solver, make sure it is supported in PythonMHD
    if constants.RIEMANN_SOLVER in simPar.keys():
        if simPar[constants.RIEMANN_SOLVER] != constants.ROE:
            raise RuntimeError("You passed an invalid value for " + constants.RIEMANN_SOLVER + " in your simulation parameters "
                               + "structure. PythonMHD currently only supports Roe's Riemann Solver algorithm. "
                               + "The Roe Riemann Solver is the default option in PythonMHD, but you can request "
                               + "it explicitly by entering \"" + constants.ROE + "\" as your riemannSolver value in simPar.")
    #Check if user has specified the specific heat ratio for the ideal gas
    if constants.GAMMA in simPar.keys():
        #Make sure the specific heat ratio is a floating point value
        if type(simPar[constants.GAMMA]) != float:
            raise RuntimeError("\nYou passed an invalid data type for " + constants.GAMMA + " in your simulation parameters "
                               + "structure.\nYou can only pass a floating point value (1 < gamma <= 5/3) for the "
                               + "specific heat ratio of the ideal gas in your simulation.")
        else:
            #Make sure that the specific heat ratio is > 1 and <= 5/3
            if simPar[constants.GAMMA] <= 1.0:
                raise RuntimeError("\nYou passed an invalid value for " + constants.GAMMA + " in your simulation parameters "
                                   + "structure.\nThe specific heat ratio for the ideal gas in your simulation "
                                   + "must be a value greater than 1.\nThe default value for gamma is "
                                   + str(constants.DEFAULT_GAMMA))
            # if simPar[constants.GAMMA] > 1.67:
            #     raise RuntimeError("\nYou passed an invalid value for " + constants.GAMMA + " in your simulation parameters "
            #                        + "structure.\nThe specific heat ratio for the ideal gas in your simulation "
            #                        + "must be a value less than or equal to 5/3.\nThe default value for gamma is "
            #                        + str(constants.DEFAULT_GAMMA))
    #Check if the user has explicitly set whether or not the simulation should use an entropy fix
    #in the Riemann solver step
    if constants.ENTROPY_FIX in simPar.keys():
        #Make sure that the entropy fix parameter is a boolean
        if type(simPar[constants.ENTROPY_FIX]) != bool:
            raise RuntimeError("\nYou passed an invalid value for " + constants.ENTROPY_FIX + " in your simulation parameters structure. "
                               + "\nThe only accepted values for entropyFix are True or False.\n\nSet " + constants.ENTROPY_FIX
                               + " to True if you want to apply an entropy fix to wavespeeds in your simulation\n(i.e., increase "
                               + "wavespeeds that are below a threshold epsilon, in order to prevent or minimize "
                               + "\n       numerical instabilities in rarefaction zones).\n\nThe default value of " + constants.ENTROPY_FIX
                               + " is False, which means that PythonMHD will not apply an entropy fix\nif you remove "
                               + constants.ENTROPY_FIX + " from your simulation parameters structure.")
        elif simPar[constants.ENTROPY_FIX]:
            #Check if the user wants to set their own epsilon (i.e., min wavespeed for which we will not apply the
            #entropy fix) value
            if constants.EPSILON in simPar.keys():
                #Make sure that epsilon is an integer or float greater than zero
                if type(simPar[constants.EPSILON]) != int and type(simPar[constants.EPSILON]) != float:
                    raise RuntimeError("\nYou passed an invalid data type for " + constants.EPSILON + " in your simulation parameters "
                                       + "structure.\nYou must pass a positive integer or floating point value for "
                                       + "the entropy fix threshold epsilon.\nThe default epsilon value in PythonMHD "
                                       + "is " + str(constants.DEFAULT_EPSILON) + ".")
                else:
                    if simPar[constants.EPSILON] <= 0.0:
                        raise RuntimeError("\nYou passed an invalid value for " + constants.EPSILON + " in your simulation parameters "
                                            + "structure.\nThe entropy fix epsilon threshold must be greater than 0.")
    #Check if the user wants to save numerical data
    if constants.SAVE_DATA in simPar.keys():
        #Make sure they have passed a boolean for the saveData flag
        if type(simPar[constants.SAVE_DATA]) != bool:
            raise RuntimeError("\nYou passed an invalid value for " + constants.SAVE_DATA + " in your simulation parameters structure. "
                               + "\nThe only accepted values for saveData are True or False.\nSet " + constants.SAVE_DATA + " to True if you "
                               + "save numerical data from your simulation.\n\nThe default value of " + constants.SAVE_DATA + " is False, "
                               + "which means that PythonMHD will\nassume that you do not need saved copies of your "
                               + "numerical data if you\nremove the " + constants.SAVE_DATA + " parameter from simPar.")
        elif simPar[constants.SAVE_DATA]:
            #If the user is not plotting data, they need to specify a saving timestep size (i.e., how much time should
            #pass in the simulation between data saves).
            notPlotting = constants.PLOT_DATA not in simPar.keys()
            if constants.PLOT_DATA in simPar.keys():
                if type(simPar[constants.PLOT_DATA]) == bool:
                    notPlotting = not simPar[constants.PLOT_DATA]
            if notPlotting and constants.SAVE_DATA_DT not in simPar.keys():
                raise RuntimeError("\nYou did not specify how frequently PythonMHD should save your numerical data. "
                                   + "\nInclude " + constants.SAVE_DATA_DT + " in your simulation parameters structure to specify how much "
                                   + "time\nshould pass in the simulation between data saving actions.\n\nIf you "
                                   + "are plotting data for the simulation, the " + constants.PLOT_DATA_DT + " parameter will be used\n"
                                   + "for the numerical outputs (i.e., numerical data will be saved whenever "
                                   + "plotting occurs)\nif you do not provide a " + constants.SAVE_DATA_DT + " output.")
            elif constants.SAVE_DATA_DT in simPar.keys():
                #Make sure that the saving timestep size is an int or a float
                if type(simPar[constants.SAVE_DATA_DT]) != float and type(simPar[constants.SAVE_DATA_DT]) != int:
                    raise RuntimeError("\nYou passed an invalid data type for " + constants.SAVE_DATA_DT + " in your simulation parameters "
                                       + "structure.\nYou can only pass an integer or a floating point value "
                                       + "for the amount of time between saving numerical outputs.")
                elif simPar[constants.SAVE_DATA_DT] <= 0:
                    #Make sure that the saving timestep is greater than zero
                    raise RuntimeError("\nYou passed an invalid value for " + constants.SAVE_DATA_DT + " in your simulation parameters "
                                       + "structure.\nThe amount of time between saving numerical outputs must be greater "
                                       + "zero.")
                #Check if the user has requested a specific format for their numerical data outputs
                if constants.SAVE_DATA_FORMAT in simPar.keys():
                    if not isinstance(simPar[constants.SAVE_DATA_FORMAT],str):
                        raise RuntimeError("\nYou have passed an invalid data type for " + constants.SAVE_DATA_FORMAT
                                           + " in your simulation parameters dictionary.\nYou must specify the "
                                           + "file format for numerical outputs by passing a string (for a valid "
                                           + "file type) for " + constants.SAVE_DATA_FORMAT + ".\nThe supported "
                                           + "numerical file types in PythonMHD are the following: "
                                           + str(constants.VALID_DATA_FILE_FORMATS)
                                           + "\n\nNote: If you select the " + constants.DATA_FORMAT_VTK
                                           + " option, PythonMHD will save your data as unstructured points vtk files, "
                                           + "\n      which is why the file extension on the outputs will be .vtu rather than .vtk.")
                    elif simPar[constants.SAVE_DATA_FORMAT] not in constants.VALID_DATA_FILE_FORMATS:
                        raise RuntimeError("\nYou have passed an unsupported file format as your " + constants.SAVE_DATA_FORMAT
                                           + " parameter in the simulation parameters dictionary.\nThe supported "
                                           + "numerical file types in PythonMHD are the following: "
                                           + str(constants.VALID_DATA_FILE_FORMATS) + ".\nIf you do not specify a "
                                           + "file format, PythonMHD will by default save your numerical data as ."
                                           + constants.DEFAULT_DATA_FORMAT + " files.\n\nNote: If you select the "
                                           + constants.DATA_FORMAT_VTK + " option, PythonMHD will save your data as "
                                           + "unstructured points vtk files,\n      which is why the file extension on the outputs "
                                           + "will be .vtu rather than .vtk.")
    #Check if the user wants to plot their simulation data
    if constants.PLOT_DATA in simPar.keys():
        #Make sure that they have passed a boolean for the plotData flag
        if type(simPar[constants.PLOT_DATA]) != bool:
            raise RuntimeError("\nYou passed an invalid data type for " + constants.PLOT_DATA + " in your simulation parameters "
                                + "structure.\nThe " + constants.PLOT_DATA + " flag can only be set as True or False.\nSet " + constants.PLOT_DATA + " "
                                + "to True if you want PythonMHD to generate any visual outputs for your simulation.")
        else:
            #If the user is plotting data, we need to know how often we should update the visualizations
            if simPar[constants.PLOT_DATA]:
                if constants.PLOT_DATA_DT not in simPar.keys():
                    raise RuntimeError("\nIf you set your " + constants.PLOT_DATA + " flag to True, you must provide a value for " + constants.PLOT_DATA_DT + " "
                                       + "in your simulation parameters structure.\nThe " + constants.PLOT_DATA_DT + " value "
                                       + "specifies how often PythonMHD will update the visualizations in simulation "
                                       + "time units \n(e.g., 0.01 for plotting every 0.01 time units, 0.1 for plotting "
                                       + "every 0.1 time units, etc.).")
                else:
                    #Make sure the plotting timestep parameter is a float or integer
                    if type(simPar[constants.PLOT_DATA_DT]) != float and type(simPar[constants.PLOT_DATA_DT]) != int:
                        raise RuntimeError("\nYou passed an invalid data type for " + constants.PLOT_DATA_DT + " in your simulation parameters "
                                           + "structure.\nYou can only pass an integer or a floating point value "
                                           + "for the amount of time between visualization outputs.")
                    else:
                        #The plotting timestep must be greater than zero
                        if simPar[constants.PLOT_DATA_DT] <= 0:
                            raise RuntimeError("\nYou passed an invalid value for " + constants.PLOT_DATA_DT + " in your simulation parameters "
                                               + "structure.\nThe amount of time between visualization outputs must "
                                               + "be greater than zero.")
                        else:
                            #We want to let the user know if their plotting timestep will produce an excessive number
                            #of visualizations (which can significantly slow down their simulation)
                            numPlotBreaks = int(simPar[constants.TLIM]/simPar[constants.PLOT_DATA_DT]) + 1
                            if numPlotBreaks > constants.MAX_PLOT_BREAKS:
                                raise RuntimeError("\nYour " + constants.PLOT_DATA_DT + " value will result in " + str(numPlotBreaks) + " plot updates. "
                                                   + "\nTo improve performance, you should increase your " + constants.PLOT_DATA_DT + " value. "
                                                   + "\n\nThe maximum number of plot updates in PythonMHD is "
                                                   + str(constants.MAX_PLOT_BREAKS) + ".\nYou can override this "
                                                   + "limit by changing MAX_PLOT_BREAKS in PythonMHD_Constants.py.")
                            #Now we should make sure that they have chosen a valid matplotlib backend (if they want to
                            #override the default Tkagg backend)
                            if constants.MATPLOTLIB_BACKEND in simPar.keys():
                                if not isinstance(simPar[constants.MATPLOTLIB_BACKEND],str):
                                    raise RuntimeError("\nYou passed an invalid data type for the " + constants.MATPLOTLIB_BACKEND
                                                       + " parameter in simPar.\nThis parameter must be a string for a "
                                                       + "valid matplotlib backend.\n\nA list of matplotlib's built-in "
                                                       + "backends is available at https://matplotlib.org/stable/users/explain/figure/backends.html. "
                                                       + "\nPlease ensure that you select one of the interactive backends (otherwise "
                                                       + "no visualizations will show up on your screen).\n\nThe default "
                                                       + "backend in PythonMHD is " + constants.DEFAULT_MATPLOTLIB_BACKEND
                                                       + ". This backend was used for all PythonMHD testing, so please "
                                                       + "be aware\nthat unexpected behaviours may occur if you switch "
                                                       + "to another matplotlib backend.")
                                elif simPar[constants.MATPLOTLIB_BACKEND] not in constants.MATPLOTLIB_INTERACTIVE_BACKENDS:
                                    raise RuntimeError("\nYou passed an invalid string for the " + constants.MATPLOTLIB_BACKEND
                                                       + " parameter in simPar.\nThis parameter must be a string for a "
                                                       + "valid matplotlib backend.\n\nA list of matplotlib's built-in "
                                                       + "backends is available at https://matplotlib.org/stable/users/explain/figure/backends.html. "
                                                       + "\nPlease ensure that you select one of the interactive backends (otherwise "
                                                       + "no visualizations will show up on your screen).\n\nThe default "
                                                       + "backend in PythonMHD is " + constants.DEFAULT_MATPLOTLIB_BACKEND
                                                       + ". This backend was used for all PythonMHD testing, so please "
                                                       + "be aware\nthat unexpected behaviours may occur if you switch "
                                                       + "to another matplotlib backend.")
                            #Check if the user wants to save their visualizations
                            if constants.SAVE_FIGS in simPar.keys():
                                #Make sure that their saveFigs flag is a boolean
                                if type(simPar[constants.SAVE_FIGS]) != bool:
                                    raise RuntimeError("\nYou passed an invalid value for " + constants.SAVE_FIGS + " in your simulation parameters structure. "
                                                       + "\n\nThe only accepted values for " + constants.SAVE_FIGS + " are True or False.\nSet " + constants.SAVE_FIGS + " to True if you "
                                                       + "save all of your visualization outputs.\n\nThe default value of " + constants.SAVE_FIGS + " is False, which "
                                                       + "which means that PythonMHD will assume\nthat you do not need saved copies of your "
                                                       + "visualizations if you remove the " + constants.SAVE_FIGS + " parameter from simPar.")
                                #If we are saving figures, we need to check whether the user has specified their output format
                                # (e.g., "png", "pdf", etc.) and/or their resolution (in dots-per-inch/dpi).
                                if simPar[constants.SAVE_FIGS]:
                                    #Check if the user wants to make a movie with their saved visualizations
                                    if constants.MAKE_MOVIE in simPar.keys():
                                        #Make sure that their makeMovie flag is a boolean
                                        if type(simPar[constants.MAKE_MOVIE]) != bool:
                                            raise RuntimeError("\nYou passed an invalid value for " + constants.MAKE_MOVIE + " in your simulation parameters structure. "
                                                               + "\n\nThe only accepted values for " + constants.MAKE_MOVIE + " are True or False.\nSet " + constants.MAKE_MOVIE + " to True if you "
                                                               + "save all of your visualization outputs.\n\nThe default value of " + constants.MAKE_MOVIE + " is False, which "
                                                               + "which means that PythonMHD will assume\nthat you do not need movies of your "
                                                               + "visualizations if you remove the " + constants.MAKE_MOVIE+ " parameter from simPar.")
                                    #For now, just make sure that the output format parameter
                                    #(if one has been supplied) is a string (whether the output
                                    #format itself is valid will depend on the matplotlib backend).
                                    #The rest of the file format verification will take place in
                                    #the SimVisualizer class.
                                    if constants.SAVE_FIGS_FORMAT in simPar.keys():
                                        if not isinstance(simPar[constants.SAVE_FIGS_FORMAT],str):
                                            raise RuntimeError("\nYou passed an invalid value for " + constants.SAVE_FIGS_FORMAT + " in your simulation "
                                                               + "parameters structure.\nYou must pass a string value for the image format parameter "
                                                               + "(e.g., \"png\", \"pdf\", etc.).\n\nThe supported file types will depend on your "
                                                               + "matplotlib backend (see PythonMHD user guide for more details).")
                                        elif constants.MAKE_MOVIE in simPar.keys():
                                            # Make sure that their makeMovie flag is a boolean
                                            if simPar[constants.MAKE_MOVIE] and simPar[constants.SAVE_FIGS_FORMAT] != constants.IMAGE_FORMAT_PNG:
                                                raise RuntimeError(
                                                    "\nYou passed an invalid value for " + constants.SAVE_FIGS_FORMAT
                                                    + " in your simulation parameters structure.\nIf you "
                                                    + "want to make movies of your simulation data, you need to"
                                                    + "save the visualization outputs as .png files.")
                                    #Make sure that the output image resolution (if one has been specified), is an
                                    #integer greater than zero. PythonMHD will warn the user, but not raise an error,
                                    #if the number of dots-per-inch is outside the recommend range.
                                    if constants.SAVE_FIGS_DPI in simPar.keys():
                                        if type(simPar[constants.SAVE_FIGS_DPI]) != int:
                                            raise RuntimeError("\nYou passed an invalid data type for " + constants.SAVE_FIGS_DPI
                                                               + " in your simulation parameters.\nThe number of dots per inch (dpi) "
                                                               + "in your output figures must be an integer value (e.g., 300, 450, etc.). "
                                                               + "\n\nIf you remove the dpi parameter from simPar, PythonMHD will use "
                                                               + "a default resolution of " + str(constants.DEFAULT_SAVE_FIGS_DPI) + ".")
                                        elif simPar[constants.SAVE_FIGS_DPI] <= 0:
                                            raise RuntimeError("\nYou passed an invalid value for " + constants.SAVE_FIGS_DPI
                                                               + " in your simulation parameters.\nThe number of dots per inch (dpi) "
                                                               + "in your output figures must be an integer value greater than zero. "
                                                               + "\n\nIf you remove the dpi parameter from simPar, PythonMHD will use "
                                                               + "a default resolution of " + str(constants.DEFAULT_SAVE_FIGS_DPI) + "."
                                                               + "\nIn most cases, you will want to use at least this resolution to "
                                                               + "produce high-quality images.")
                                        elif simPar[constants.SAVE_FIGS_DPI] < constants.MIN_RECOMMENDED_SAVE_FIGS_DPI:
                                            raise RuntimeError("\nYou passed a value for " + constants.SAVE_FIGS_DPI
                                                               + " in your simulation parameters that is below the minimum "
                                                               + "\nrecommended resolution for output images ("
                                                               + str(constants.MIN_RECOMMENDED_SAVE_FIGS_DPI) + " dots per inch).\n\n"
                                                               + "If you remove the dpi parameter from simPar, PythonMHD will use "
                                                               + "a default resolution of " + str(constants.DEFAULT_SAVE_FIGS_DPI) + "."
                                                               + "\nIn most cases, you will want to use at least this resolution to "
                                                               + "produce high-quality images.")
                                        elif simPar[constants.SAVE_FIGS_DPI] > constants.MAX_RECOMMENDED_SAVE_FIGS_DPI:
                                            raise RuntimeError("\nYou passed a value for " + constants.SAVE_FIGS_DPI
                                                               + " in your simulation parameters that is above the maximum "
                                                               + "\nrecommended resolution for output images ("
                                                               + str(constants.MAX_RECOMMENDED_SAVE_FIGS_DPI) + " dots per inch)."
                                                               + "\n\nIf you remove the dpi parameter from simPar, PythonMHD will use "
                                                               + "a default resolution of " + str(constants.DEFAULT_SAVE_FIGS_DPI) + "."
                                                               + "\nIn most cases, a resolution closer to this value will be "
                                                               + "sufficient to produce high-quality images.\nIf you make the dpi too large, "
                                                               + "the output images will take up an excessive amount of space on your computer.")

    #Determine whether PythonMHD will be saving any files to the user's machine
    savingFiles = constants.SAVE_FIGS in simPar.keys() or constants.SAVE_DATA in simPar.keys()
    if constants.SAVE_FIGS in simPar.keys():
        savingFiles = simPar[constants.SAVE_FIGS]
    if not savingFiles and constants.SAVE_DATA in simPar.keys():
        savingFiles = simPar[constants.SAVE_DATA]
    #If they are saving data or figures, they might have specified a path to where the outputs folder should be created,
    #a name for the outputs folder, and a prefix name for all of the output files
    if savingFiles:
        #If the user has specified a path for where to create the outputs folder, make sure the parameter is a string
        #and that the path actually exists on the user's machine
        if constants.OUTPUT_FOLDER_PATH in simPar.keys():
            if not isinstance(simPar[constants.OUTPUT_FOLDER_PATH], str):
                raise RuntimeError("\nYou passed an invalid value for " + constants.OUTPUT_FOLDER_PATH + " in your simulation parameters structure. "
                                   + "\nIf you want to specify the location where the simulation outputs folder should be "
                                   + "created,\nyou must pass a string for the " + constants.OUTPUT_FOLDER_PATH + " parameter.")
            elif not os.path.exists(simPar[constants.OUTPUT_FOLDER_PATH]):
                raise RuntimeError("\nThe path that you provided for the " + constants.OUTPUT_FOLDER_PATH + " parameter in simPar does not "
                                   + "exist on your machine.\nPlease provide a valid path for the desired location "
                                   + "of your simulation outputs, or remove " + constants.OUTPUT_FOLDER_PATH + "\nfrom simPar if you "
                                   + "want to save your data in PythonMHD's \"Outputs\" folder.")
        #Make sure the output folder name is a string with a reasonable number of characters
        if constants.OUTPUT_FOLDER_NAME in simPar.keys():
            if not isinstance(simPar[constants.OUTPUT_FOLDER_NAME], str):
                raise RuntimeError("\nYou passed an invalid value for " + constants.OUTPUT_FOLDER_NAME + " in your simulation parameters structure. "
                                   + "\nIf you want to specify a name for the simulation outputs folder, you must pass "
                                   + "a string\nfor the " + constants.OUTPUT_FOLDER_NAME + " parameter.")
            elif len(simPar[constants.OUTPUT_FOLDER_NAME].strip()) == 0:
                raise RuntimeError("\nYou passed an invalid string for the  " + constants.OUTPUT_FOLDER_NAME
                                   + " parameter in your simPar dictionary.\nThe output folder name that you passed "
                                   + "is either an empty string or only contains whitespace characters.")
            elif len(simPar[constants.OUTPUT_FOLDER_NAME]) > constants.MAX_OUTPUT_FOLDER_NAME_CHARACTERS:
                raise RuntimeError("\nYou passed a string that is too long for the  " + constants.OUTPUT_FOLDER_NAME
                                   + " parameter in your simPar dictionary.\nThe maximum length of your output folder name "
                                   + "is " + str(constants.MAX_OUTPUT_FOLDER_NAME_CHARACTERS) + " characters.\n\nYou can override "
                                   + "this value by changing MAX_OUTPUT_FOLDER_NAME_CHARACTERS in the PythonMHD_Constants.py file.")
        #Make sure the output file name is a string with a reasonable number of characters
        if constants.OUTPUT_FILE_NAME in simPar.keys():
            if not isinstance(simPar[constants.OUTPUT_FILE_NAME], str):
                raise RuntimeError("\nYou passed an invalid value for " + constants.OUTPUT_FILE_NAME + " in your simulation parameters structure. "
                                   + "\nIf you want to specify a prefix for simulation output files, you must pass "
                                   + "a string\nfor the " + constants.OUTPUT_FILE_NAME + " parameter.")
            elif len(simPar[constants.OUTPUT_FILE_NAME]) > constants.MAX_OUTPUT_FILE_NAME_CHARACTERS:
                raise RuntimeError("\nYou passed a string that is too long for the " + constants.OUTPUT_FILE_NAME
                                   + " parameter in your simPar dictionary.\nThe maximum length of the output file name "
                                   + "is " + str(constants.MAX_OUTPUT_FILE_NAME_CHARACTERS) + " characters.\nYou can override "
                                   + "this value by changing MAX_OUTPUT_FILE_NAME_CHARACTERS in the PythonMHD_Constants.py file.")









