#EvolveFunctions_Hydro.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides functions for evolving a 1D, 2D, or 3D hydrodynamic simulation by a requested amount of time.
#Additional Information: All of the evolve functions in PythonMHD are Godunov-based (i.e., they evolve the state of
#                        a hydrodynamic system by using Godunov's scheme [1]). In order to briefly explain how
#                        Godunov's scheme works, let's consider a three dimensional grid cell at the indices
#                        (y_index = j, x_index = i, z_index = k). If U^t_j,i,k is the conservative variable state
#                        in cell (j,i ,k) at time t, we want to find the new conservative variable state at time
#                        t + dt, U^t+dt_j,i,k. If F_j,i-1/2,k is the flux through the left edge of cell (j,i,k),
#                        F_j,i+1/2,k is the flux through the right edge, G_j-1/2,i,k is the flux through the top edge,
#                        G_j+1/2,i,k is the flux through the bottom edge, H_j,i,k-1/2 is the flux through the back
#                        edge, and H_j,i,k+1/2 is the flux through the front/forward edge, we can calculate U^t+dt_j,i,k
#                        with the following formula, which is the 3D version of Godunov's scheme:
#                        U^t+dt_j,i,k = U^t_j,i,k - (dt/dx)*(F_j,i+1/2,k - F_j,i-1/2,k)
#                                                 - (dt/dy)*(G_j+1/2,i,k - G_j-1/2,i,k)
#                                                 - (dt/dz)*(H_j,i,k+1/2 - H_j,i,k-1/2)
#                        See the recommend textbook chapter by Toro (2009) [3] for a more detailed description of
#                        Godunov's scheme for finite grid simulations (i.e., simulations where we divide a continuous
#                        gas system into a finite number of cells/pixels that each contain their own values for density,
#                        velocities, and pressure).
#
#                        Godunov's scheme alone is sufficient for evolving a 1D hydrodynamic system, but we use
#                        the Corner Transport Upwind (CTU) method proposed in (Collela, 1990) [2] to improve the
#                        effective spatial resolution of PythonMHD for 2D and 3D simulations. Instead of immediately
#                        putting our flux vectors for all two or three spatial dimensions into Godunov's formula
#                        (i.e., the directionally split approach, because all spatial dimensions are treated
#                        independently), we perform Collela's half-timestep correction that takes into account
#                        transverse flux contributions (i.e., flux contributions along the other one or two
#                        spatial dimensions). This approach is known as a directionally unsplit algorithm because
#                        the calculations that we perform in one spatial dimension depend on the results
#                        that we derive for the other spatial dimensions. After calculating the intercell flux
#                        values at time t, we evolve the reconstructed left/right, top/bottom, and back/forward
#                        primitive variable states (i.e., the inputs that we pass to the Riemann solver that calculates
#                        the intercell flux values) by half a timestep using the transverse flux values. Below are the
#                        equations that we use for these transverse updates in 2D and 3D.
#                        2D Transverse Flux Half Dt Update (X-Direction)
#                        Visualization:
#                         _ _ __G_j-1/2,i-1,k _ _ _ _ _ _G_j-1/2,i,k _ _ _
#                        |                       |                       |
#                        |                       |                       |
#                        |        U_L_(j,i-1/2,k)|U_R_(j,i-1/2,k)        |
#                        |                       |                       |
#                        |                       |                       |
#                        | _ _ _G_j+1/2,i-1,k_ _ | _ _ _G_j+1/2,i,k _ _ _|
#                        G_j-1/2,i-1,k = upper-left y-direction intercell flux
#                        G_j+1/2,i-1,k = lower-left y-direction intercell flux
#                        G_j-1/2,i,k = upper-right y-direction intercell flux
#                        G_j+1/2,i,k = lower-right y-direction intercell flux
#                        Formulas:
#                        Left Conservative Variables: U_L_(j,i-1/2,k) -= 0.5*(dt/dy)*(G_j+1/2,i-1,k - G_j-1/2,i-1,k)
#                        Right Conservative Variables: U_R_(j,i-1/2,k) -= 0.5*(dt/dy)*(G_j+1/2,i,k - G_j-1/2,i,k)
#                        [Note the similarity to Godunov's scheme. When we make the transverse corrections, we apply
#                        Godunov's scheme using only transverse flux contributions (i.e., we leave out the fluxes
#                        that are in the same direction as the reconstructed states we are currently updating).]
#
#                        2D Transverse Flux Half Dt Update (Y-Direction)
#                        Visualization:
#                         _ _ _ _ _ _ _ _ _ _ _ _
#                        |                       |
#                        |                       |
#                        |                       |
#                   F_j-1,i-1/2,k          F_j-1,i+1/2,k
#                        |                       |
#                        |                       |
#                        |     U_T_(j-1/2,i,k)   |
#                        |- - - - - - - - - - - -|
#                        |     U_B_(j-1/2,i,k)   |
#                        |                       |
#                        |                       |
#                   F_j,i-1/2,k             F_j,i+1/2,k
#                        |                       |
#                        |                       |
#                        |_ _ _ _ _ _ _ _ _ _ _ _|
#                        F_j-1,i-1/2,k = upper-left x-direction intercell flux
#                        F_j-1,i+1/2,k = upper-right x-direction intercell flux
#                        F_j,i-1/2,k = lower-left x-direction intercell flux
#                        F_j,i+1/2,k = lower-right x-direction intercell flux
#                        Formulas:
#                        Top Conservative Variables: U_T_(j-1/2,i,k) -= 0.5*(dt/dx)*(F_j-1,i+1/2,k - F_j-1,i-1/2,k)
#                        Bottom Conservative Variables: U_B_(j-1/2,i,k) -= 0.5*(dt/dx)*(F_j,i+1/2,k - F_j,i-1/2,k)
#
#                        3D Transverse Flux Half Dt Update (X-Direction)
#                        [Visualizations are not provided for the 3D updates because it is difficult to clearly
#                         present all six sides of the cell in this format. Instead, please refer to PythonMHD's
#                         user guide if you would like to see visualizations of the 3D transverse updates.]
#                        [Important Note: In 3D simulations, the positive z-direction (i.e., the "forward" direction
#                         in PythonMHD's terminology) is into the screen, whereas the negative z-direction (i.e.,
#                         the "backward" direction) is out of the screen.]
#                        Flux Definitions:
#                        G_j-1/2,i-1,k = upper-left y-direction intercell flux
#                        G_j+1/2,i-1,k = lower-left y-direction intercell flux
#                        G_j-1/2,i,k = upper-right y-direction intercell flux
#                        G_j+1/2,i,k = lower-right y-direction intercell flux
#                        H_j,i-1,k-1/2 = back-left z-direction intercell flux
#                        H_j,i-1,k+1/2 = forward-left z-direction intercell flux
#                        H_j,i,k-1/2 = back-right z-direction intercell flux
#                        H_j,i,k+1/2 = forward-right z-direction intercell flux
#                        Update Formulas:
#                        Left Conservative Variables: U_L_(j,i-1/2,k) -= 0.5*(dt/dy)*(G_j+1/2,i-1,k - G_j-1/2,i-1,k)
#                                                                      + 0.5*(dt/dz)*(H_j,i-1,k+1/2 - H_j,i-1,k-1/2)
#                        Right Conservative Variables: U_R_(j,i-1/2,k) -= 0.5*(dt/dy)*(G_j+1/2,i,k - G_j-1/2,i,k)
#                                                                       + 0.5*(dt/dz)*(H_j,i,k+1/2 - H_j,i,k-1/2)
#
#                        3D Transverse Flux Half Dt Update (Y-Direction)
#                        Flux Definitions:
#                        F_j-1,i-1/2,k = upper-left x-direction intercell flux
#                        F_j-1,i+1/2,k = upper-right x-direction intercell flux
#                        F_j,i-1/2,k = lower-left x-direction intercell flux
#                        F_j,i+1/2,k = lower-right x-direction intercell flux
#                        H_j-1,i,k-1/2 = upper-back z-direction intercell flux
#                        H_j-1,i,k+1/2 = upper-forward z-direction intercell flux
#                        H_j,i,k-1/2 = lower-back z-direction intercell flux
#                        H_j,i,k+1/2 = lower-forward z-direction intercell flux
#                        Update Formulas:
#                        Top Conservative Variables: U_T_(j-1/2,i,k) -= 0.5*(dt/dx)*(F_j-1,i+1/2,k - F_j-1,i-1/2,k)
#                                                                     + 0.5*(dt/dz)*(H_j-1,i,k+1/2 - H_j-1,i,k-1/2)
#                        Bottom Conservative Variables: U_B_(j-1/2,i,k) -= 0.5*(dt/dx)*(F_j,i+1/2,k - F_j,i-1/2,k)
#                                                                        + 0.5*(dt/dz)*(H_j,i,k+1/2 - H_j,i,k-1/2)
#
#                        3D Transverse Flux Half Dt Update (Z-Direction)
#                        Flux Definitions:
#                        F_j,i-1/2,k-1 = back-left x-direction intercell flux
#                        F_j,i+1/2,k-1 = back-right x-direction intercell flux
#                        F_j,i-1/2,k = forward-left x-direction intercell flux
#                        F_j,i+1/2,k = forward-right x-direction intercell flux
#                        G_j-1/2,i,k-1 = back-upper y-direction intercell flux
#                        G_j+1/2,i,k-1 = back-lower y-direction intercell flux
#                        G_j-1/2,i,k = forward-upper y-direction intercell flux
#                        G_j+1/2,i,k = forward-lower y-direction intercell flux
#                        Update Formulas:
#                        Back Conservative Variables: U_Bk_(j,i,k-1/2) -= 0.5*(dt/dx)*(F_j,i+1/2,k-1 - F_j,i-1/2,k-1)
#                                                                       + 0.5*(dt/dy)*(G_j+1/2,i,k-1 - G_j-1/2,i,k-1)
#                        Forward Conservative Variables: U_F_(j,i,k-1/2) -= 0.5*(dt/dx)*(F_j,i+1/2,k - F_j,i-1/2,k)
#                                                                         + 0.5*(dt/dy)*(G_j+1/2,i,k - G_j-1/2,i,k)
#
#                        After we have these half-timestep reconstructed states, we recalculate all of the intercell
#                        flux vectors, using all of the new reconstructed states that we just derived. These new fluxes
#                        are what we put into Godunov's scheme to update our cell-centred primitive variables by a
#                        full timestep.
#                        U^t+dt_j,i,k = U^t_j,i,k - (dt/dx)*(F^t+dt/2_j,i+1/2,k - F^t+dt/2_j,i-1/2,k)
#                                                 - (dt/dy)*(G^t+dt/2_j+1/2,i,k - G^t+dt/2_j-1/2,i,k)
#                                                 - (dt/dz)*(H^t+dt/2_j,i,k+1/2 - H^t+dt/2_j,i,k-1/2)
#
#                        This implementation of Colella's CTU method is designed to generate results that are numerically
#                        identical (within 10^(-15) - 10^(-13), see PythonMHD user guide for additional information on
#                        numerical differences between C and Python) to those generated by the 2017 version of Athena [4].
#                        The original methods paper for Athena (Stone et al., 2008) [5] is another excellent resource
#                        for understanding how the CTU method is used in computational hydrodynamics and
#                        magnetohydrodynamics.
#
#References:
# 1. Godunov, S. K., & Bohachevsky, I. (1959). Finite difference method for numerical calculation of
#    discontinuous solutions of the equations of hydrodynamics. Mat. Sb. (N.S.), 47(3), 271-306.
#    http://mi.mathnet.ru/eng/msb4873.
# 2. Colella, P. (1990). Multidimensional upwind methods for hyperbolic conservation laws. Journal
#    of Computational Physics, 87(1), 171-200. https://doi.org/10.1016/0021-9991(90)90233-Q.
# 3. Toro, E. F. (2009). The method of Godunov for non-linear systems. In: Riemann solvers and
#    numerical methods for fluid dynamics: A practical introduction. Springer, Berlin, Heidelberg.
#    https://doi-org.uml.idm.oclc.org/10.1007/b79761_6.
# 4. https://github.com/PrincetonUniversity/Athena-Cversion
# 5. Stone, J. M., Gardiner, T. A., Teuben, P., Hawley, J. F., & Simon, J. B. (2008).
#    Athena: A new code for astrophysical MHD. The Astrophysical Journal Supplemental Series,
#    178(1), 137-177. https://iopscience.iop.org/article/10.1086/588755/pdf.


#####IMPORT STATEMENTS#####

#Import helper module for Primitive -> Conservative and Conservative -> Primitive variable conversions
import Source.PrimCons_HelperModule as primConsLib

#Import helper module for primitive variable reconstruction
import Source.Reconstruction_PPM_Hydro_HelperModule as ppmLib

#Import helper module for intercell flux calculations
import Source.RiemannSolvers_Hydro_Roe_HelperModule as roeLib

#Import helper module for timestep size calculations
import Source.TimestepCalculations_HelperModule as dtLib

#Import helper module for PythonMHD constants
import Source.PythonMHD_Constants as constants

#Import NumPy for matrix operations
import numpy as np

#####1D HYDRODYNAMIC EVOLVE FUNCTION#####

#Function: evolve_1D_hydro
#Purpose: Simulates the time evolution of a 1D hydrodynamic simulation from
#         t = tStart to t = tEnd. When it reaches the specified time limit, it
#         returns the new cell-centred primitive and conservative variable states
#         for the entire simulation grid.
#         Note: This function should only be used on Cartesian simulation grids.
#Input Parameters: simulationGrid (the Simulation Grid object, which contains the
#                                  initial state of the system, boundary conditions,
#                                  and the size of each grid cell)
#                  tStart (the time at which we are starting the time evolution)
#                  tEnd (the time at which we must end the time evolution)
#                  startCycle (the cycle number we are starting at)
#                             (Because evolve functions can be called multiple times
#                              during the same simulation, we need a start cycle number
#                              so that the user never sees the cycle numbers reset
#                              between evolve function calls.)
#                  maxCycles (the max cycle number that we can reach in pursuit
#                             of tEnd) (If we reach the maxCycles value, we will
#                             return our last calculated state for the primitive
#                             variables and provide the time at which we stopped
#                             the simulation)
#                  gamma (the specific heat ratio of the ideal gas)
#                  cfl (the CFL number for timestep size calculations)
#                  minDens (the minimum density value for the simulation)
#                  minPres (the minimum pressure value for the simulation)
#                  minEnergy (the minimum energy value for the simulation)
#                  reconstructOrder (the spatial reconstruction order we should apply
#                                    (0 = none, 1 = plm, 2 = ppm), 0 by default)
#                  entropyFix (boolean flag for whether we should apply an entropy fix
#                              during the intercell flux calculations (see the
#                              RiemannSolvers_Hydro_Roe_HelperModule.py script in Source
#                              if you want information on the purpose of entropy fixes
#                              and whether using one would be beneficial in your simulation).
#                  epsilon (epsilon value for the entropy fix, 0.5 by default)
#Outputs: primVars (the primitive variable states at time t = tEnd (or the primitive
#                   variable states at the moment when we reached maxCycles))
#         consVars (the conservative variable states at time t = tEnd (or the primitive
#                   variable states at the moment when we reached maxCycles))
#         time (the new time in the simulation (should equal tEnd if we didn't reach maxCycles))
#         cycle (the new cycle number for the simulation)
#         secondLastPrimVars (the state of the primitive variables at the end of the second-last cycle)
#         secondLastConsVars (the state of the conservative variables at the end of the second-last cycle)
#         secondLastTime (the time at the end of the second-last cycle)
def evolve_1D_hydro(simulationGrid, tStart, tEnd, startCycle, maxCycles, gamma, cfl, minDens, minPres,
                    minEnergy, reconstructOrder=0, entropyFix=False, epsilon=0.5):
    #Retrieve the primitive variables from the simulation grid
    primVars = simulationGrid.primVars
    #Ensure that the simulation grid is 1D
    assert(len(primVars.shape) == 2)
    #Retrieve the conservative variables from the simulation grid
    consVars = simulationGrid.consVars
    assert(primVars.shape == consVars.shape)
    #Retrieve the cell width from the simulation grid
    dx = simulationGrid.dx
    assert(dx > 0)
    #Retrieve the boundary condition from the simulation grid
    BcX = simulationGrid.BcX
    assert(BcX == constants.OUTFLOW or BcX == constants.PERIODIC)
    #Create a variable for keeping track of the simulation time
    time = tStart
    assert(tEnd >= tStart)
    #Create a variable for keeping track of the cycle number
    cycle = startCycle
    assert(startCycle >= 0 and maxCycles > 0)
    #Initialize the second-last primitive variables
    secondLastPrimVars = np.copy(primVars)
    #Initialize the second-last conservative variables
    secondLastConsVars = np.copy(consVars)
    #Initialize the second-last time
    secondLastTime = tStart
    #Keep iterating until we reach tEnd or the max cycle number
    while time < tEnd and cycle < maxCycles:
        #Update the second-last variables and time
        secondLastPrimVars = np.copy(primVars)
        secondLastConsVars = np.copy(consVars)
        secondLastTime = time
        #Get the timestep size for this cycle
        dt = dtLib.calcDt_1D_hydro(consVars,gamma,cfl,dx)
        #Make sure we don't go over tEnd
        if time + dt > tEnd:
            dt = tEnd - time
        #Start the cycle by performing spatial reconstruction
        #Check which spatial reconstruction order the user requested
        if reconstructOrder == constants.PPM_SPATIAL_RECONSTRUCTION: #if we should use PPM reconstruction
            (leftPrimVars, rightPrimVars) = ppmLib.ppmReconstructX_hydro(primVars, gamma, dt, dx, BcX)
        # elif reconstructOrder == 1: #if we should use PLM reconstruction
        #     (leftPrimVars, rightPrimVars) = plmLib.ppmReconstruct1D_hydro(primVars, gamma, dt, dx, BcX)
        else: #if we shouldn't perform any spatial reconstruction
            #Apply boundary conditions to determine the correct left and right
            #variable states at the edges of the grid
            if BcX == 0: #if the BC is outflow
                leftPrimVars = np.append(primVars[:,0].reshape(-1,1),primVars[:,:],axis=1)
                rightPrimVars = np.append(primVars[:,:],primVars[:,primVars.shape[1]-1].reshape(-1,1),axis=1)
            else: #if the BC is periodic
                leftPrimVars = np.append(primVars[:,primVars.shape[1]-1].reshape(-1,1),primVars[:,:],axis=1)
                rightPrimVars = np.append(primVars[:,:],primVars[:,0].reshape(-1,1),axis=1)
        #Apply density and pressure floors to the reconstructed states
        leftPrimVars = primConsLib.applyFloors(leftPrimVars,True,minDens,minPres,minEnergy,
                                                "the x-direction reconstruction outputs (left)")
        rightPrimVars = primConsLib.applyFloors(rightPrimVars,True,minDens,minPres,minEnergy,
                                                "the x-direction reconstruction outputs (right)")
        #Find the intercell fluxes between the left and right states
        interCellFlux = roeLib.riemannSolverX_hydro_roe(leftPrimVars, rightPrimVars,gamma,entropyFix, epsilon)
        leftInterCellFlux = interCellFlux[:,0:interCellFlux.shape[1]-1]
        rightInterCellFlux = interCellFlux[:,1:interCellFlux.shape[1]]
        #Update the conservative variables with Godunov's scheme
        dtdx = dt/dx
        consVars -= dtdx * (rightInterCellFlux - leftInterCellFlux)
        #Calculate the new primitive variable statues
        primVars[:,:] = primConsLib.consToPrim_hydro_x(consVars, gamma)
        time += dt
        cycle += 1
        #Print the time, cycle number, and timestep size
        print("Cycle = " + str(cycle) + " Time = " + str(time) + " Dt = " + str(dt))
    #Return the final state of the primitive + conservative variables and the time
    #and cycle when we stopped the time evolution
    return primVars, consVars, time, cycle, secondLastPrimVars, secondLastConsVars, secondLastTime

#Function: evolve_2D_hydro
#Purpose: Simulates the time evolution of a 2D hydrodynamic simulation from
#         t = tStart to t = tEnd, using the CTU method [2]. When the evolve function
#         reaches the specified time limit, it returns the new cell-centred
#         primitive and conservative variable states for the entire simulation grid.
#         Note: This function should only be used on Cartesian simulation grids.
#Input Parameters: simulationGrid (the Simulation Grid object, which contains the
#                                  initial state of the system, boundary conditions,
#                                  and the size of each grid cell)
#                  tStart (the time at which we are starting the time evolution)
#                  tEnd (the time at which we must end the time evolution)
#                  startCycle (the cycle number we are starting at)
#                             (Because evolve functions can be called multiple times
#                              during the same simulation, we need a start cycle number
#                              so that the user never sees the cycle numbers reset
#                              between evolve function calls.)
#                  maxCycles (the max cycle number that we can reach in pursuit
#                             of tEnd) (If we reach the maxCycles value, we will
#                             return our last calculated state for the primitive
#                             variables and provide the time at which we stopped
#                             the simulation)
#                  gamma (the specific heat ratio of the ideal gas)
#                  cfl (the CFL number for timestep size calculations)
#                  minDens (the minimum density value for the simulation)
#                  minPres (the minimum pressure value for the simulation)
#                  minEnergy (the minimum energy value for the simulation)
#                  reconstructOrder (the spatial reconstruction order we should apply
#                                    (0 = none, 2 = ppm), 0 by default)
#                  entropyFix (boolean flag for whether we should apply an entropy fix
#                              during the intercell flux calculations (see the
#                              RiemannSolvers_Hydro_Roe_HelperModule.py script in Source
#                              if you want information on the purpose of entropy fixes
#                              and whether using one would be beneficial in your simulation).
#                  epsilon (epsilon value for the entropy fix, 0.5 by default)
#Outputs: primVars (the primitive variable states at time t = tEnd (or the primitive
#                   variable states at the moment when we reached maxCycles))
#         consVars (the conservative variable states at time t = tEnd (or the primitive
#                   variable states at the moment when we reached maxCycles))
#         time (the new time in the simulation (should equal tEnd if we didn't reach maxCycles))
#         cycle (the new cycle number for the simulation)
#         secondLastPrimVars (the state of the primitive variables at the end of the second-last cycle)
#         secondLastConsVars (the state of the conservative variables at the end of the second-last cycle)
#         secondLastTime (the time at the end of the second-last cycle)
def evolve_2D_hydro(simulationGrid, tStart, tEnd, startCycle, maxCycles, gamma, cfl, minDens, minPres, minEnergy,
                    reconstructOrder=0, entropyFix=False, epsilon=0.5):
    #Retrieve the primitive variables from the simulation grid
    primVars = simulationGrid.primVars
    #Ensure that the simulation grid is 3D
    assert(len(primVars.shape) == 3)
    #Retrieve the conservative variables from the simulation grid
    consVars = simulationGrid.consVars
    assert(primVars.shape == consVars.shape)
    #Retrieve the number of cells in the x-direction
    numXCells = simulationGrid.numXCells
    assert(numXCells > 1)
    #Retrieve the number of cells in the y-direction
    numYCells = simulationGrid.numYCells
    assert(numYCells > 1)
    #Retrieve the cell width from the simulation grid
    dx = simulationGrid.dx
    assert(dx > 0)
    #Retrieve the cell height from the simulation grid
    dy = simulationGrid.dy
    assert(dy > 0)
    #Retrieve the boundary condition for the x-direction
    BcX = simulationGrid.BcX
    assert(BcX == constants.OUTFLOW or BcX == constants.PERIODIC)
    #Retrieve the boundary condition for the y-direction
    BcY = simulationGrid.BcY
    assert(BcY == constants.OUTFLOW or BcY == constants.PERIODIC)
    #Create a variable for keeping track of the simulation time
    time = tStart
    assert(tStart <= tEnd)
    #Create a variable for keeping track of the cycle number
    cycle = startCycle
    assert(startCycle >= 0 and maxCycles > 0)
    #Initialize the second-last primitive variables
    secondLastPrimVars = np.copy(primVars)
    #Initialize the second-last conservative variables
    secondLastConsVars = np.copy(consVars)
    #Initialize the second-last time
    secondLastTime = tStart
    #Keep iterating until we reach tEnd or the max cycle number
    while time < tEnd and cycle < maxCycles:
        #Update the second-last variables and time
        secondLastPrimVars = np.copy(primVars)
        secondLastConsVars = np.copy(consVars)
        secondLastTime = time
        #Get the timestep size for this cycle
        dt = dtLib.calcDt_2D_hydro(consVars, gamma, cfl, dx, dy)
        #Make sure we don't go over tEnd
        if time + dt > tEnd:
            dt = tEnd - time
        #Calculate some timestep-related values that will be helpful
        #for Godunov's scheme and the CTU method
        dtdx = dt/dx
        dtdy = dt/dy
        halfDtDx = 0.5*dtdx
        halfDtDy = 0.5*dtdy
        #Start the cycle by performing spatial reconstruction
        #Check which spatial reconstruction order the user requested
        if reconstructOrder == 2: #if we should use PPM reconstruction
            (leftPrimVars, rightPrimVars) = ppmLib.ppmReconstructX_hydro(primVars, gamma, dt, dx, BcX)
            (topPrimVars, bottomPrimVars) = ppmLib.ppmReconstructY_hydro(primVars, gamma, dt, dy, BcY)
        # elif reconstructOrder == 1: #if we should use PLM reconstruction
        #     (leftPrimVars, rightPrimVars) = plmLib.ppmReconstruct1D_hydro(primVars, gamma, dt, dx, BcX)
        else: #if we shouldn't perform any spatial reconstruction
            #Apply boundary conditions to determine the correct left and right
            #variable states at the edges of the grid
            if BcX == 0: #if the BC is outflow
                leftPrimVars = np.append(primVars[:,:,0].reshape(primVars.shape[0],-1,1),
                                         primVars[:,:,:],axis=2)
                rightPrimVars = np.append(primVars[:,:,:],
                                          primVars[:,:,primVars.shape[2]-1].reshape(primVars.shape[0],-1,1),axis=2)
            else: #if the BC is periodic
                leftPrimVars = np.append(primVars[:,:,primVars.shape[2]-1].reshape(primVars.shape[0],-1,1),
                                         primVars[:,:,:],axis=2)
                rightPrimVars = np.append(primVars[:,:,:],
                                          primVars[:,:,0].reshape(primVars.shape[0],-1,1),axis=2)
            if BcY == 0: #if the BC is outflow
                topPrimVars = np.append(primVars[:,0,:].reshape(primVars.shape[0],1,-1),
                                        primVars[:,:,:],axis=1)
                bottomPrimVars = np.append(primVars[:,:,:],
                                           primVars[:,primVars.shape[1]-1,:].reshape(primVars.shape[0],1,-1),axis=1)
            else: #if the BC is periodic
                topPrimVars = np.append(primVars[:,primVars.shape[1]-1,:].reshape(primVars.shape[0],1,-1),
                                        primVars[:,:,:],axis=1)
                bottomPrimVars = np.append(primVars[:,:,:],
                                           primVars[:,0,:].reshape(primVars.shape[0],1,-1),axis=1)
        #Apply density and pressure floors to the reconstructed states
        leftPrimVars = primConsLib.applyFloors(leftPrimVars,True,minDens,minPres,minEnergy,
                                                "the x-direction reconstruction outputs (left)")
        rightPrimVars = primConsLib.applyFloors(rightPrimVars,True,minDens,minPres,minEnergy,
                                                "the x-direction reconstruction outputs (right)")
        topPrimVars = primConsLib.applyFloors(topPrimVars,True,minDens,minPres,minEnergy,
                                                "the y-direction reconstruction outputs (top)")
        bottomPrimVars = primConsLib.applyFloors(bottomPrimVars,True,minDens,minPres,minEnergy,
                                                "the y-direction reconstruction outputs (bottom)")
        #Get the conservative form of the reconstructed states
        leftConsVars = primConsLib.primToCons_hydro_x(leftPrimVars, gamma)
        rightConsVars = primConsLib.primToCons_hydro_x(rightPrimVars, gamma)
        topConsVars = primConsLib.primToCons_hydro_y(topPrimVars, gamma)
        bottomConsVars = primConsLib.primToCons_hydro_y(bottomPrimVars, gamma)

        #Find the intercell fluxes between the left and right states
        intercellFluxX = roeLib.riemannSolverX_hydro_roe(leftPrimVars, rightPrimVars, gamma, entropyFix, epsilon)
        leftIntercellFlux = intercellFluxX[:,:,0:intercellFluxX.shape[2]-1]
        rightIntercellFlux = intercellFluxX[:,:,1:intercellFluxX.shape[2]]
        #Find the intercell fluxes between the top and bottom states
        intercellFluxY = roeLib.riemannSolverY_hydro_roe(topPrimVars, bottomPrimVars, gamma, entropyFix, epsilon)
        topIntercellFlux = intercellFluxY[:,0:intercellFluxY.shape[1]-1,:]
        bottomIntercellFlux = intercellFluxY[:,1:intercellFluxY.shape[1],:]

        #In accordance with the CTU method [2], we will now use transverse flux contributions to evolve the
        #reconstructed states by half a timestep. For the left and right states, we use the vertical/y-direction
        #flux vectors. For the top and bottom states, we instead use the horizontal/x-direction fluxes for the
        #half-timestep update.
        if BcX == constants.OUTFLOW:
            leftIntercellFluxY = np.append(intercellFluxY[:,:,0].reshape(5,-1,1), intercellFluxY, axis=2)
            rightIntercellFluxY = np.append(intercellFluxY, intercellFluxY[:,:,numXCells-1].reshape(5,-1,1), axis=2)
        else:
            leftIntercellFluxY = np.append(intercellFluxY[:,:,numXCells-1].reshape(5,-1,1), intercellFluxY, axis=2)
            rightIntercellFluxY = np.append(intercellFluxY, intercellFluxY[:,:,0].reshape(5,-1,1), axis=2)
        if BcY == constants.OUTFLOW:
            topIntercellFluxX = np.append(intercellFluxX[:,0,:].reshape(5,1,-1), intercellFluxX, axis=1)
            bottomIntercellFluxX = np.append(intercellFluxX, intercellFluxX[:,numYCells-1,:].reshape(5,1,-1), axis=1)
        else:
            topIntercellFluxX = np.append(intercellFluxX[:,numYCells-1,:].reshape(5,1,-1), intercellFluxX, axis=1)
            bottomIntercellFluxX = np.append(intercellFluxX, intercellFluxX[:,0,:].reshape(5,1,-1), axis=1)
        leftConsVars -= halfDtDy*(leftIntercellFluxY[:,1:intercellFluxY.shape[1],:]
                                  - leftIntercellFluxY[:,0:intercellFluxY.shape[1]-1,:])
        rightConsVars -= halfDtDy*(rightIntercellFluxY[:,1:intercellFluxY.shape[1],:]
                                  - rightIntercellFluxY[:,0:intercellFluxY.shape[1]-1,:])
        topConsVars -= halfDtDx*(topIntercellFluxX[:,:,1:intercellFluxX.shape[2]]
                                  - topIntercellFluxX[:,:,0:intercellFluxX.shape[2]-1])
        bottomConsVars -= halfDtDx*(bottomIntercellFluxX[:,:,1:intercellFluxX.shape[2]]
                                  - bottomIntercellFluxX[:,:,0:intercellFluxX.shape[2]-1])

        #Get the primitive form of the updated reconstructed states
        leftPrimVars = primConsLib.consToPrim_hydro_x(leftConsVars, gamma)
        rightPrimVars = primConsLib.consToPrim_hydro_x(rightConsVars, gamma)
        topPrimVars = primConsLib.consToPrim_hydro_y(topConsVars, gamma)
        bottomPrimVars = primConsLib.consToPrim_hydro_y(bottomConsVars, gamma)

        #Find the new intercell fluxes between the left and right states
        newIntercellFluxX = roeLib.riemannSolverX_hydro_roe(leftPrimVars, rightPrimVars, gamma, entropyFix, epsilon)
        leftIntercellFlux = newIntercellFluxX[:,:,0:intercellFluxX.shape[2]-1]
        rightIntercellFlux = newIntercellFluxX[:,:,1:intercellFluxX.shape[2]]
        #Find the new intercell fluxes between the top and bottom states
        newIntercellFluxY = roeLib.riemannSolverY_hydro_roe(topPrimVars, bottomPrimVars, gamma, entropyFix, epsilon)
        topIntercellFlux = newIntercellFluxY[:,0:intercellFluxY.shape[1]-1,:]
        bottomIntercellFlux = newIntercellFluxY[:,1:intercellFluxY.shape[1],:]

        #Update the conservative variables with Godunov's scheme,
        #using the new intercell flux values
        consVars -= dtdx*(rightIntercellFlux - leftIntercellFlux)
        consVars -= dtdy*(bottomIntercellFlux - topIntercellFlux)

        #Calculate the new cell-centred primitive variable statues
        primVars[:,:] = primConsLib.consToPrim_hydro_x(consVars, gamma)

        #Update the time and cycle number
        time += dt
        cycle += 1

        #Print the time, cycle number, and timestep size
        print("Cycle = " + str(cycle) + " Time = " + str(time) + " Dt = " + str(dt))
    #Return the final state of the primitive + conservative variables and the time
    #and cycle when we stopped the time evolution
    return primVars, consVars, time, cycle, secondLastPrimVars, secondLastConsVars, secondLastTime

#Function: evolve_3D_hydro
#Purpose: Simulates the time evolution of a 3D hydrodynamic simulation from
#         t = tStart to t = tEnd, using the CTU method [2]. When the evolve function
#         reaches the specified time limit, it returns the new cell-centred
#         primitive and conservative variable states for the entire simulation grid.
#         Note: This function should only be used on Cartesian simulation grids.
#Input Parameters: simulationGrid (the Simulation Grid object, which contains the
#                                  initial state of the system, boundary conditions,
#                                  and the size of each grid cell)
#                  tStart (the time at which we are starting the time evolution)
#                  tEnd (the time at which we must end the time evolution)
#                  startCycle (the cycle number we are starting at)
#                             (Because evolve functions can be called multiple times
#                              during the same simulation, we need a start cycle number
#                              so that the user never sees the cycle numbers reset
#                              between evolve function calls.)
#                  maxCycles (the max cycle number that we can reach in pursuit
#                             of tEnd) (If we reach the maxCycles value, we will
#                             return our last calculated state for the primitive
#                             variables and provide the time at which we stopped
#                             the simulation)
#                  gamma (the specific heat ratio of the ideal gas)
#                  cfl (the CFL number for timestep size calculations)
#                  minDens (the minimum density value for the simulation)
#                  minPres (the minimum pressure value for the simulation)
#                  minEnergy (the minimum energy value for the simulation)
#                  reconstructOrder (the spatial reconstruction order we should apply
#                                    (0 = none, 2 = ppm), 0 by default)
#                  entropyFix (boolean flag for whether we should apply an entropy fix
#                              during the intercell flux calculations (see the
#                              RiemannSolvers_Hydro_Roe_HelperModule.py script in Source
#                              if you want information on the purpose of entropy fixes
#                              and whether using one would be beneficial in your simulation).
#                  epsilon (epsilon value for the entropy fix, 0.5 by default)
#Outputs: primVars (the primitive variable states at time t = tEnd (or the primitive
#                      variable states at the moment when we reached maxCycles))
#         time (the new time in the simulation (should equal tEnd if we didn't reach maxCycles))
#         cycle (the new cycle number for the simulation)
#         secondLastPrimVars (the state of the primitive variables at the end of the second-last cycle)
#         secondLastConsVars (the state of the conservative variables at the end of the second-last cycle)
#         secondLastTime (the time at the end of the second-last cycle)
def evolve_3D_hydro(simulationGrid, tStart, tEnd, startCycle, maxCycles, gamma, cfl, minDens, minPres,
                    minEnergy, reconstructOrder=0, entropyFix=False, epsilon=0.5):
    #Retrieve the primitive variables from the simulation grid
    primVars = simulationGrid.primVars
    #Ensure that the simulation grid is 3D
    assert(len(primVars.shape) - 1 == 3)
    #Retrieve the conservative variables from the simulation grid
    consVars = simulationGrid.consVars
    assert(primVars.shape == consVars.shape)
    #Retrieve the number of cells in the x-direction
    numXCells = simulationGrid.numXCells
    assert(numXCells > 1)
    #Retrieve the number of cells in the y-direction
    numYCells = simulationGrid.numYCells
    assert(numYCells > 1)
    #Retrieve the number of cells in the z-direction
    numZCells = simulationGrid.numZCells
    assert(numZCells > 1)
    #Retrieve the cell width from the simulation grid
    dx = simulationGrid.dx
    assert(dx > 0)
    #Retrieve the cell height from the simulation grid
    dy = simulationGrid.dy
    assert(dy > 0)
    #Retrieve the cell depth from the simulation grid
    dz = simulationGrid.dz
    assert(dz > 0)
    #Retrieve the boundary condition for the x-direction
    BcX = simulationGrid.BcX
    assert(BcX == constants.OUTFLOW or BcX == constants.PERIODIC)
    #Retrieve the boundary condition for the y-direction
    BcY = simulationGrid.BcY
    assert(BcY == constants.OUTFLOW or BcY == constants.PERIODIC)
    #Retrieve the boundary condition for the z-direction
    BcZ = simulationGrid.BcZ
    assert(BcZ == constants.OUTFLOW or BcZ == constants.PERIODIC)
    #Create a variable for keeping track of the simulation time
    time = tStart
    assert(tStart <= tEnd)
    #Create a variable for keeping track of the cycle number
    cycle = startCycle
    assert(startCycle >= 0 and maxCycles > 0)
    #Initialize the second-last primitive variables
    secondLastPrimVars = np.copy(primVars)
    #Initialize the second-last conservative variables
    secondLastConsVars = np.copy(consVars)
    #Initialize the second-last time
    secondLastTime = tStart
    #Keep iterating until we reach tEnd or the max cycle number
    while time < tEnd and cycle < maxCycles:
        #Update the second-last variables and time
        secondLastPrimVars = np.copy(primVars)
        secondLastConsVars = np.copy(consVars)
        secondLastTime = time
        #Get the timestep size for this cycle
        dt = dtLib.calcDt_3D_hydro(consVars, gamma, cfl, dx, dy, dz)
        #Make sure we don't go over tEnd
        if time + dt > tEnd:
            dt = tEnd - time
        #Calculate some timestep-related values that will be helpful
        #for Godunov's scheme and the CTU method
        dtdx = dt/dx
        dtdy = dt/dy
        dtdz = dt/dz
        halfDtDx = 0.5*dtdx
        halfDtDy = 0.5*dtdy
        halfDtDz = 0.5*dtdz
        #Start the cycle by performing spatial reconstruction
        #Check which spatial reconstruction order the user requested
        if reconstructOrder == 2: #if we should use PPM reconstruction
            (leftPrimVars, rightPrimVars) = ppmLib.ppmReconstructX_hydro(primVars, gamma, dt, dx, BcX)
            (topPrimVars, bottomPrimVars) = ppmLib.ppmReconstructY_hydro(primVars, gamma, dt, dy, BcY)
            (backPrimVars, forwPrimVars) = ppmLib.ppmReconstructZ_hydro(primVars, gamma, dt, dz, BcZ)
        # elif reconstructOrder == 1: #if we should use PLM reconstruction
        #     (leftPrimVars, rightPrimVars) = plmLib.ppmReconstruct1D_hydro(primVars, gamma, dt, dx, BcX)
        else: #if we shouldn't perform any spatial reconstruction
            #Apply boundary conditions to determine the correct left and right
            #variable states at the edges of the grid
            if BcX == 0: #if the BC is outflow
                leftPrimVars = np.append(primVars[:,:,0,:].reshape(primVars.shape[0],primVars.shape[1],1,-1),
                                         primVars[:,:,:,:],axis=2)
                rightPrimVars = np.append(primVars[:,:,:,:],
                                          primVars[:,:,primVars.shape[2]-1,:].reshape(primVars.shape[0],
                                                                                      primVars.shape[1],1,-1),axis=2)
            else: #if the BC is periodic
                leftPrimVars = np.append(primVars[:,:,primVars.shape[2]-1,:].reshape(primVars.shape[0],
                                                                                     primVars.shape[1],1,-1),
                                         primVars[:,:,:,:],axis=2)
                rightPrimVars = np.append(primVars[:,:,:,:],
                                          primVars[:,:,0,:].reshape(primVars.shape[0],primVars.shape[1],1,-1),axis=2)
            if BcY == 0: #if the BC is outflow
                topPrimVars = np.append(primVars[:,0,:,:].reshape(primVars.shape[0],1,primVars.shape[2],-1),
                                        primVars[:,:,:,:],axis=1)
                bottomPrimVars = np.append(primVars[:,:,:,:],
                                           primVars[:,primVars.shape[1]-1,:,:].reshape(primVars.shape[0],1,
                                                                                       primVars.shape[2],-1),axis=1)
            else: #if the BC is periodic
                topPrimVars = np.append(primVars[:,primVars.shape[1]-1,:,:].reshape(primVars.shape[0],1,
                                                                                    primVars.shape[2],-1),
                                        primVars[:,:,:,:],axis=1)
                bottomPrimVars = np.append(primVars[:,:,:,:],
                                           primVars[:,0,:,:].reshape(primVars.shape[0],1,
                                                                     primVars.shape[2],-1),axis=1)
            if BcZ == 0: #if the BC is outflow
                backPrimVars = np.append(primVars[:,:,:,0].reshape(primVars.shape[0],primVars.shape[1],-1,1),
                                         primVars[:,:,:,:],axis=3)
                forwPrimVars = np.append(primVars[:,:,:,:],
                                          primVars[:,:,:,primVars.shape[3]-1].reshape(primVars.shape[0],
                                                                                      primVars.shape[1],-1,1),axis=3)
            else: #if the BC is periodic
                backPrimVars = np.append(primVars[:,:,:,primVars.shape[3]-1].reshape(primVars.shape[0],
                                                                                     primVars.shape[1],-1,1),
                                         primVars[:,:,:,:],axis=3)
                forwPrimVars = np.append(primVars[:,:,:,:],
                                          primVars[:,:,:,0].reshape(primVars.shape[0],primVars.shape[1],-1,1),axis=3)
        #Apply density and pressure floors to the reconstructed states
        leftPrimVars = primConsLib.applyFloors(leftPrimVars,True,minDens,minPres,minEnergy,
                                                "the x-direction reconstruction outputs (left)")
        rightPrimVars = primConsLib.applyFloors(rightPrimVars,True,minDens,minPres,minEnergy,
                                                "the x-direction reconstruction outputs (right)")
        topPrimVars = primConsLib.applyFloors(topPrimVars,True,minDens,minPres,minEnergy,
                                                "the y-direction reconstruction outputs (top)")
        bottomPrimVars = primConsLib.applyFloors(bottomPrimVars,True,minDens,minPres,minEnergy,
                                                "the y-direction reconstruction outputs (bottom)")
        backPrimVars = primConsLib.applyFloors(backPrimVars,True,minDens,minPres,minEnergy,
                                                "the z-direction reconstruction outputs (back)")
        forwPrimVars = primConsLib.applyFloors(forwPrimVars,True,minDens,minPres,minEnergy,
                                                "the z-direction reconstruction outputs (forward)")
        #Get the conservative form of the reconstructed states
        leftConsVars = primConsLib.primToCons_hydro_x(leftPrimVars, gamma)
        rightConsVars = primConsLib.primToCons_hydro_x(rightPrimVars, gamma)
        topConsVars = primConsLib.primToCons_hydro_y(topPrimVars, gamma)
        bottomConsVars = primConsLib.primToCons_hydro_y(bottomPrimVars, gamma)
        backConsVars = primConsLib.primToCons_hydro_z(backPrimVars, gamma)
        forwConsVars = primConsLib.primToCons_hydro_z(forwPrimVars, gamma)

        #Find the intercell fluxes between the left and right states
        intercellFluxX = roeLib.riemannSolverX_hydro_roe(leftPrimVars, rightPrimVars, gamma, entropyFix, epsilon)
        leftIntercellFlux = intercellFluxX[:,:,0:intercellFluxX.shape[2]-1,:]
        rightIntercellFlux = intercellFluxX[:,:,1:intercellFluxX.shape[2],:]
        #Find the intercell fluxes between the top and bottom states
        intercellFluxY = roeLib.riemannSolverY_hydro_roe(topPrimVars, bottomPrimVars, gamma, entropyFix, epsilon)
        topIntercellFlux = intercellFluxY[:,0:intercellFluxY.shape[1]-1,:,:]
        bottomIntercellFlux = intercellFluxY[:,1:intercellFluxY.shape[1],:,:]
        #Find the intercell fluxes between the backward and forward states
        intercellFluxZ = roeLib.riemannSolverZ_hydro_roe(backPrimVars, forwPrimVars, gamma, entropyFix, epsilon)
        backIntercellFlux = intercellFluxZ[:,:,:,0:intercellFluxZ.shape[3]-1]
        forwIntercellFlux = intercellFluxZ[:,:,:,1:intercellFluxZ.shape[3]]

        #In accordance with the CTU method [2], we will now use transverse flux contributions to evolve the
        #reconstructed states by half a timestep. For the left and right states, we use the vertical/y-direction
        #and depth/z-direction flux vectors. For the top and bottom states, we use the horizontal/x-direction and
        #depth/z-direction fluxes for the half-timestep update. For the back and forward states, we use the
        #horizontal/x-direction and vertical/y-direction fluxes.
        if BcX == constants.OUTFLOW:
            leftIntercellFluxY = np.append(intercellFluxY[:,:,0,:].reshape(5,numYCells+1,1,-1), intercellFluxY, axis=2)
            rightIntercellFluxY = np.append(intercellFluxY, intercellFluxY[:,:,numXCells-1,:].reshape(5,numYCells+1,1,-1), axis=2)
            leftIntercellFluxZ = np.append(intercellFluxZ[:,:,0,:].reshape(5,numYCells,1,-1), intercellFluxZ, axis=2)
            rightIntercellFluxZ = np.append(intercellFluxZ, intercellFluxZ[:,:,numXCells-1,:].reshape(5,numYCells+1,1,-1), axis=2)
        else:
            leftIntercellFluxY = np.append(intercellFluxY[:,:,numXCells-1,:].reshape(5,numYCells+1,1,-1), intercellFluxY, axis=2)
            rightIntercellFluxY = np.append(intercellFluxY, intercellFluxY[:,:,0,:].reshape(5,numYCells+1,1,-1), axis=2)
            leftIntercellFluxZ = np.append(intercellFluxZ[:,:,numXCells-1,:].reshape(5,numYCells,1,-1), intercellFluxZ, axis=2)
            rightIntercellFluxZ = np.append(intercellFluxZ, intercellFluxZ[:,:,0,:].reshape(5,numYCells,1,-1), axis=2)
        if BcY == constants.OUTFLOW:
            topIntercellFluxX = np.append(intercellFluxX[:,0,:,:].reshape(5,1,numXCells+1,-1), intercellFluxX, axis=1)
            bottomIntercellFluxX = np.append(intercellFluxX, intercellFluxX[:,numYCells-1,:,:].reshape(5,1,numXCells+1,-1), axis=1)
            topIntercellFluxZ = np.append(intercellFluxZ[:,0,:,:].reshape(5,1,numXCells,-1), intercellFluxZ, axis=1)
            bottomIntercellFluxZ = np.append(intercellFluxZ, intercellFluxZ[:,numYCells-1,:,:].reshape(5,1,numXCells,-1), axis=1)
        else:
            topIntercellFluxX = np.append(intercellFluxX[:,numYCells-1,:,:].reshape(5,1,numXCells+1,-1), intercellFluxX, axis=1)
            bottomIntercellFluxX = np.append(intercellFluxX, intercellFluxX[:,0,:,:].reshape(5,1,numXCells+1,-1), axis=1)
            topIntercellFluxZ = np.append(intercellFluxZ[:,numYCells-1,:,:].reshape(5,1,numXCells,-1), intercellFluxZ, axis=1)
            bottomIntercellFluxZ = np.append(intercellFluxZ, intercellFluxZ[:,0,:,:].reshape(5,1,numXCells,-1), axis=1)
        if BcZ == constants.OUTFLOW:
            backIntercellFluxX = np.append(intercellFluxX[:,:,:,0].reshape(5,numYCells,-1,1), intercellFluxX, axis=3)
            forwIntercellFluxX = np.append(intercellFluxX, intercellFluxX[:,:,:,numZCells-1].reshape(5,numYCells,-1,1), axis=3)
            backIntercellFluxY = np.append(intercellFluxY[:,:,:,0].reshape(5,numYCells+1,-1,1), intercellFluxY, axis=3)
            forwIntercellFluxY = np.append(intercellFluxY, intercellFluxY[:,:,:,numZCells-1].reshape(5,numYCells+1,-1,1), axis=3)
        else:
            backIntercellFluxX = np.append(intercellFluxX[:,:,:,numZCells-1].reshape(5,numYCells,-1,1), intercellFluxX, axis=3)
            forwIntercellFluxX = np.append(intercellFluxX, intercellFluxX[:,:,:,0].reshape(5,numYCells,-1,1), axis=3)
            backIntercellFluxY = np.append(intercellFluxY[:,:,:,numZCells-1].reshape(5,numYCells+1,-1,1), intercellFluxY, axis=3)
            forwIntercellFluxY = np.append(intercellFluxY, intercellFluxY[:,:,:,0].reshape(5,numYCells+1,-1,1), axis=3)
        leftConsVars -= halfDtDy*(leftIntercellFluxY[:,1:intercellFluxY.shape[1],:,:]
                                  - leftIntercellFluxY[:,0:intercellFluxY.shape[1]-1,:,:])
        leftConsVars -= halfDtDz*(leftIntercellFluxZ[:,:,:,1:intercellFluxZ.shape[3]]
                                  - leftIntercellFluxZ[:,:,:,0:intercellFluxZ.shape[3]-1])
        rightConsVars -= halfDtDy*(rightIntercellFluxY[:,1:intercellFluxY.shape[1],:,:]
                                  - rightIntercellFluxY[:,0:intercellFluxY.shape[1]-1,:,:])
        rightConsVars -= halfDtDz*(rightIntercellFluxZ[:,:,:,1:intercellFluxZ.shape[3]]
                                  - rightIntercellFluxZ[:,:,:,0:intercellFluxZ.shape[3]-1])
        topConsVars -= halfDtDx*(topIntercellFluxX[:,:,1:intercellFluxX.shape[2],:]
                                  - topIntercellFluxX[:,:,0:intercellFluxX.shape[2]-1,:])
        topConsVars -= halfDtDz*(topIntercellFluxZ[:,:,:,1:intercellFluxZ.shape[3]]
                                  - topIntercellFluxZ[:,:,:,0:intercellFluxZ.shape[3]-1])
        bottomConsVars -= halfDtDx*(bottomIntercellFluxX[:,:,1:intercellFluxX.shape[2],:]
                                  - bottomIntercellFluxX[:,:,0:intercellFluxX.shape[2]-1,:])
        bottomConsVars -= halfDtDz*(bottomIntercellFluxZ[:,:,:,1:intercellFluxZ.shape[3]]
                                  - bottomIntercellFluxZ[:,:,:,0:intercellFluxZ.shape[3]-1])
        backConsVars -= halfDtDx*(backIntercellFluxX[:,:,1:intercellFluxX.shape[2],:]
                                  - backIntercellFluxX[:,:,0:intercellFluxX.shape[2]-1,:])
        backConsVars -= halfDtDy*(backIntercellFluxY[:,1:intercellFluxY.shape[1],:,:]
                                  - backIntercellFluxY[:,0:intercellFluxY.shape[1]-1,:,:])
        forwConsVars -= halfDtDx*(forwIntercellFluxX[:,:,1:intercellFluxX.shape[2],:]
                                  - forwIntercellFluxX[:,:,0:intercellFluxX.shape[2]-1,:])
        forwConsVars -= halfDtDy*(forwIntercellFluxY[:,1:intercellFluxY.shape[1],:,:]
                                  - forwIntercellFluxY[:,0:intercellFluxY.shape[1]-1,:,:])

        #Get the primitive form of the updated reconstructed states
        leftPrimVars = primConsLib.consToPrim_hydro_x(leftConsVars, gamma)
        rightPrimVars = primConsLib.consToPrim_hydro_x(rightConsVars, gamma)
        topPrimVars = primConsLib.consToPrim_hydro_y(topConsVars, gamma)
        bottomPrimVars = primConsLib.consToPrim_hydro_y(bottomConsVars, gamma)
        backPrimVars = primConsLib.consToPrim_hydro_z(backConsVars, gamma)
        forwPrimVars = primConsLib.consToPrim_hydro_z(forwConsVars, gamma)

        #Find the new intercell fluxes between the left and right states
        newIntercellFluxX = roeLib.riemannSolverX_hydro_roe(leftPrimVars, rightPrimVars, gamma, entropyFix, epsilon)
        leftIntercellFlux = newIntercellFluxX[:,:,0:intercellFluxX.shape[2]-1,:]
        rightIntercellFlux = newIntercellFluxX[:,:,1:intercellFluxX.shape[2],:]
        #Find the new intercell fluxes between the top and bottom states
        newIntercellFluxY = roeLib.riemannSolverY_hydro_roe(topPrimVars, bottomPrimVars, gamma, entropyFix, epsilon)
        topIntercellFlux = newIntercellFluxY[:,0:intercellFluxY.shape[1]-1,:,:]
        bottomIntercellFlux = newIntercellFluxY[:,1:intercellFluxY.shape[1],:,:]
        #Find the new intercell fluxes between the back and forward states
        newIntercellFluxZ = roeLib.riemannSolverZ_hydro_roe(backPrimVars, forwPrimVars, gamma, entropyFix, epsilon)
        backIntercellFlux = newIntercellFluxZ[:,:,:,0:intercellFluxZ.shape[3]-1]
        forwIntercellFlux = newIntercellFluxZ[:,:,:,1:intercellFluxZ.shape[3]]

        #Update the conservative variables with Godunov's scheme,
        #using the new intercell flux values
        consVars -= dtdx*(rightIntercellFlux - leftIntercellFlux)
        consVars -= dtdy*(bottomIntercellFlux - topIntercellFlux)
        consVars -= dtdz*(forwIntercellFlux - backIntercellFlux)

        #Calculate the new cell-centred primitive variable statues
        primVars[:,:] = primConsLib.consToPrim_hydro_x(consVars, gamma)

        #Update the time and cycle number
        time += dt
        cycle += 1

        #Print the time, cycle number, and timestep size
        print("Cycle = " + str(cycle) + " Time = " + str(time) + " Dt = " + str(dt))
    #Return the final state of the primitive + conservative variables and
    #the time and cycle when we stopped the time evolution
    return primVars, consVars, time, cycle, secondLastPrimVars, secondLastConsVars, secondLastTime
