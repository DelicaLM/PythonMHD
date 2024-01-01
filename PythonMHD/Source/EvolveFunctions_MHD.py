#EvolveFunctions_MHD.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides functions for evolving a 1D, 2D, or 3D magnetohydrodynamic simulation by a requested amount of time.
#
#Additional Information: All of the evolve functions in PythonMHD are Godunov-based (i.e., they evolve the state of
#                        a magnetohydrodynamic system by using Godunov's scheme [2]). In order to briefly explain how
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
#                        Godunov's scheme alone is sufficient for evolving a 1D MHD system, but we a more complicated
#                        algorithm to enforce the divergence-free condition on magnetic fields in 2D and 3D. In 1D
#                        simulations, we easily maintain the divergence-free condition by simply ensuring that the
#                        x-component of the magnetic field has the same value in every cell (Bx = const ->
#                        div_1D(B) = dBx/dx = d(const)/dx = 0). In 2D and 3D, PythonMHD uses Athena's Corner Transport
#                        Upwind + Constrained Transport (CTU + CT) method [4,5,6] to maintain the divergence-free condition
#                        within numerical error. This algorithm combines Colella's CTU algorithm (1990) [7] for hydroydnamics
#                        (see EvolveFunctions_Hydro.py for more details) with Evans' and Hawley's CT approach [8] for MHD
#                        simulations. The CT part of the algorithm means that we calculate the new cell-centred magnetic
#                        fields at the end of each simulation cycle by taking the average of adjacent face-centred magnetic
#                        field values.
#                        CT Magnetic Field Updates
#                        X-Direction
#                        _ _ _ _ _ _ _ _ _ _ _ _ _
#                        |                       |
#                        |                       |
#                   faceBx_i-1/2  centBx_i  faceBx_i+1/2
#                        |                       |
#                        |                       |
#                        | _ _ _ _ _ _ _ _ _ _ _ |
#                                 Cell i
#                        centBx^(t+dt)_i = 0.5*(faceBx^(t+dt)_(i-1/2) + faceBx^(t+dt)_i+1/2)
#
#                        Y-Direction
#                        - - - - faceBy_j-1/2- - -
#                        |                       |
#                        |                       |
#                        |        centBy_j       |
#                        |                       |
#                        |                       |
#                        - - - - faceBy_j+1/2- - -
#                        centBy^(t+dt)_j = 0.5*(faceBy^(t+dt)_(j-1/2) + faceBy^(t+dt)_j+1/2)
#
#                        Z-Direction
#                        [A visualization is not provided for the z-direction, because it is difficult to show the
#                         third spatial dimension in this format.]
#                        centBz^(t+dt)_j = 0.5*(faceBz^(t+dt)_(k-1/2) + faceBz^(t+dt)_k+1/2)
#
#                        In order to calculate the face-centred magnetic fields at time t + dt, we need to use the
#                        electric field values at the cell corners in our simulation grid. Even though electric fields
#                        are not included in our primitive or conservative variable sets, we can easily calculate the
#                        cell-centred electric field values by taking the cross product of velocity and the magnetic
#                        field (-v x B). The complicated part is figuring out what the electric field is at the corners
#                        of a cell, because we don't know the velocity and magnetic field values at these locations.
#                        As a solution for this problem, PythonMHD uses the averaging formulas described in (Gardiner
#                        & Stone, 2005) [6] and (Stone et al., 2008) [5] to find the corner electric field/electromotive
#                        force (EMF) values. These formulas are described in PythonMHD's "CornerEMF_HelperModule.Py"
#                        script. Once we have the corner electric field/EMF values, we use the formulas below (see [5]
#                        for details on why these formulas maintain the divergence-free condition) to evolve the
#                        face-centred magnetic fields.
#
#                        2D Face-Centred Magnetic Field Updates
#                        j = y-index, i = x-index
#                        faceBx^(t+dt)_(j,i-1/2) = faceBx^(t)_(j,i) - (dt/dy)*(cornerEMF_z^(t+dt/2)_(j+1/2,i-1/2)
#                                                                               - cornerEMF_z^(t+dt/2)_(j-1/2,i-1/2))
#                        faceBy^(t+dt)_(j-1/2,i) = faceBy^(t)_(j,i) + (dt/dx)*(cornerEMF_z^(t+dt/2)_(j-1/2,i+1/2)
#                                                                               - cornerEMF_z^(t+dt/2)_(j-1/2,i-1/2))
#
#                        3D Face-Centred Magnetic Field Updates
#                        j = y-index, i = x-index, k = z-index
#                        faceBx^(t+dt)_(j,i-1/2,k) = faceBx^(t)_(j,i-1/2,k) - (dt/dy)*(cornerEMF_z^(t+dt/2)_(j+1/2,i-1/2,k)
#                                                                                      - cornerEMF_z^(t+dt/2)_(j-1/2,i-1/2,k))
#                                                                           + (dt/dz)*(cornerEMF_y^(t+dt/2)_(j,i-1/2,k+1/2)
#                                                                                      - cornerEMF_y^(t+dt/2)_(j,i-1/2,k-1/2)
#                        faceBy^(t+dt)_(j-1/2,i,k) = faceBy^(t)_(j-1/2,i,k) + (dt/dx)*(cornerEMF_z^(t+dt/2)_(j-1/2,i+1/2,k)
#                                                                                      - cornerEMF_z^(t+dt/2)_(j-1/2,i-1/2,k))
#                                                                           - (dt/dz)*(cornerEMF_x^(t+dt/2)_(j-1/2,i,k+1/2)
#                                                                                       - cornerEMF_x^(t+dt/2)_(j-1/2,i,k-1/2)
#                        faceBz^(t+dt)_(j,i,k-1/2) = faceBz^(t)_(j,i,k-1/2) - (dt/dx)*(cornerEMF_y^(t+dt/2)_(j+1/2,i,k-1/2)
#                                                                                       - cornerEMF_y^(t+dt/2)_(j-1/2,i,k-1/2))
#                                                                           + (dt/dy)*(cornerEMF_x^(t+dt/2)_(j+1/2,i,k-1/2)
#                                                                                       - cornerEMF_x^(t+dt/2)_(j-1/2,i,k-1/2)
#
#
#References:
# 1. Colella, P. (1990). Multidimensional upwind methods for hyperbolic conservation laws. Journal
#    of Computational Physics, 87(1), 171-200. https://doi.org/10.1016/0021-9991(90)90233-Q
# 2. Godunov, S. K., & Bohachevsky, I. (1959). Finite difference method for numerical calculation of
#    discontinuous solutions of the equations of hydrodynamics. Mat. Sb. (N.S.), 47(3), 271-306.
#    http://mi.mathnet.ru/eng/msb4873
# 3. Toro, E. F. (2009). The method of Godunov for non-linear systems. In: Riemann solvers and
#    numerical methods for fluid dynamics: A practical introduction. Springer, Berlin, Heidelberg.
#    https://doi-org.uml.idm.oclc.org/10.1007/b79761_6
# 4. https://github.com/PrincetonUniversity/Athena-Cversion
# 5. Stone, J. M., Gardiner, T. A., Teuben, P., Hawley, J. F., & Simon, J. B. (2008).
#    Athena: A new code for astrophysical MHD. The Astrophysical Journal Supplemental Series,
#    178(1), 137-177. https://iopscience.iop.org/article/10.1086/588755/pdf
# 6. Gardiner, T. A., & Stone, J. M. (2005). An unsplit Godunov method for ideal MHD via Constrained Transport.
#    Journal of Computational Physics, 205(2), 509-539. https://arxiv.org/pdf/astro-ph/0501557.pdf
# 7. Colella, P. (1990). Multidimensional upwind methods for hyperbolic conservation laws. Journal
#    of Computational Physics, 87(1), 171-200. https://doi.org/10.1016/0021-9991(90)90233-Q
# 8. Evans, C. R., & Hawley, J. F. (1988). Simulation of magnetohydrodynamic flows: A Constrained Transport method.
#    Astrophysical Journal, 332, 659-677. https://adsabs.harvard.edu/full/1988ApJ...332..659E

#####IMPORT STATEMENTS#####

#Import helper module for Primitive -> Conservative and Conservative -> Primitive variable conversions
import Source.PrimCons_HelperModule as primConsLib

#Import helper module for primitive variable reconstruction
import Source.Reconstruction_PPM_MHD_HelperModule as ppmLib

#Import helper module for intercell flux calculations
import Source.RiemannSolvers_MHD_Roe_HelperModule as roeLib

#Import helper module for timestep size calculations
import Source.TimestepCalculations_HelperModule as dtLib

#Import helper module for CT magnetic field updates (i.e., when we calculate new
#cell-centred magnetic fields from the face-centred magnetic field values)
import Source.CT_HelperModule as ctLib

#Import helper module for calculating the corner EMF values
import Source.CornerEMF_HelperModule as emfLib

#Import helper module for the CTU algorithm (i.e., the half-timestep updates on the reconstructed primitive
#variable states (see the "EvolveFunctions_Hydro.py" script for additional information on why we need to
#evolve the primitive variable states by half a timestep))
import Source.CTU_HelperModule as ctuLib

#Import helper module for PythonMHD constants
import Source.PythonMHD_Constants as constants

#Import NumPy for matrix operations
import numpy as np

#####1D MHD EVOLVE FUNCTION#####

#Function: evolve_1D_mhd
#Purpose: Simulates the time evolution of a 1D magnetohydrodynamic simulation from
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
#                                    (0 = none, 2 = ppm), 0 by default)
#                  entropyFix (boolean flag for whether we should apply an entropy fix
#                              during the intercell flux calculations (see the
#                              RiemannSolvers_MHD_Roe_HelperModule.py script in Source
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
def evolve_1D_mhd(simulationGrid, tStart, tEnd, startCycle, maxCycles, gamma, cfl, minDens, minPres,
                    minEnergy, reconstructOrder=0, entropyFix=False, epsilon=0.5):
    #Retrieve the primitive variables from the simulation grid
    primVarsAll = simulationGrid.primVars
    #Ensure that the simulation grid is 1D
    assert(len(primVarsAll.shape) == 2)
    #Ensure that 8 state variables have been defined for each cell
    assert(primVarsAll.shape[0] == 8)
    #Ensure that the number of cells is greater than zero
    assert(primVarsAll.shape[1] > 0)
    #Retrieve the Bx value for the simulation
    #(Note: Bx must be uniform in a 1D sim
    #       to maintain the divergence-free
    #       constraint.)
    Bx = primVarsAll[4][0]*np.ones(shape=primVarsAll.shape[1])
    #Make another version of Bx with an extra column
    #(for the reconstruction algorithm)
    faceBx = primVarsAll[4][0]*np.ones(shape=primVarsAll.shape[1]+1)
    #Retrieve the conservative variables from the simulation grid
    consVarsAll = simulationGrid.consVars
    assert(primVarsAll.shape == consVarsAll.shape)
    #Create a primitive variables matrix for the x-direction
    #(i.e., include all primitive variables except Bx)
    primVarsX = np.array([primVarsAll[0],
                          primVarsAll[1],
                          primVarsAll[2],
                          primVarsAll[3],
                          primVarsAll[5],
                          primVarsAll[6],
                          primVarsAll[7]])
    #Create a conservative variables matrix for the x-direction
    consVarsX = np.array([consVarsAll[0],
                          consVarsAll[1],
                          consVarsAll[2],
                          consVarsAll[3],
                          consVarsAll[5],
                          consVarsAll[6],
                          consVarsAll[7]])
    #Retrieve the cell width from the simulation grid
    dx = simulationGrid.dx
    assert(dx > 0)
    #Retrieve the boundary condition from the simulation grid
    BcX = simulationGrid.BcX
    assert(BcX == constants.OUTFLOW or BcX == constants.PERIODIC)
    #Create a variable for keeping track of the simulation time
    time = tStart
    assert(tStart <= tEnd)
    #Create a variable for keeping track of the cycle number
    cycle = startCycle
    assert(startCycle >= 0)
    #Initialize the second-last primitive variables
    secondLastPrimVars = np.copy(primVarsAll)
    #Initialize the second-last conservative variables
    secondLastConsVars = np.copy(consVarsAll)
    #Initialize the second-last time
    secondLastTime = tStart
    #Keep iterating until we reach tEnd or the max cycle number
    while time < tEnd and cycle < maxCycles:
        #Update the second-last variables and time
        secondLastPrimVars = np.copy(primVarsAll)
        secondLastConsVars = np.copy(consVarsAll)
        secondLastTime = time
        #Get the timestep size for this cycle
        dt = dtLib.calcDt_1D_mhd(consVarsAll,gamma,cfl,dx)
        #Make sure we don't go over tEnd
        if time + dt > tEnd:
            dt = tEnd - time
        #Start the cycle by performing spatial reconstruction
        #Check which spatial reconstruction order the user requested
        if reconstructOrder == 2: #if we should use PPM reconstruction
            (leftPrimVars, rightPrimVars) = ppmLib.ppmReconstructX_mhd(primVarsX, Bx, gamma, dt, dx, BcX)
        # elif reconstructOrder == 1: #if we should use PLM reconstruction
        #     (leftPrimVars, rightPrimVars) = plmLib.ppmReconstruct1D_hydro(primVars, gamma, dt, dx, BcX)
        else: #if we shouldn't perform any spatial reconstruction
            #Apply boundary conditions to determine the correct left and right
            #variable states at the edges of the grid
            if BcX == 0: #if the BC is outflow
                leftPrimVars = np.append(primVarsX[:,0].reshape(-1,1),primVarsX[:,:],axis=1)
                rightPrimVars = np.append(primVarsX[:,:], primVarsX[:,primVarsX.shape[1]-1].reshape(-1,1),axis=1)
            else: #if the BC is periodic
                leftPrimVars = np.append(primVarsX[:,primVarsX.shape[1]-1].reshape(-1,1),primVarsX[:,:],axis=1)
                rightPrimVars = np.append(primVarsX[:,:], primVarsX[:,0].reshape(-1,1),axis=1)
        #Apply density and pressure floors to the reconstructed states
        leftPrimVars = primConsLib.applyFloors(leftPrimVars, True, minDens, minPres, minEnergy,
                                               "the x-direction reconstruction outputs (left)")
        rightPrimVars = primConsLib.applyFloors(rightPrimVars, True, minDens, minPres, minEnergy,
                                               "the x-direction reconstruction outputs (right)")
        #Get the conservative form of the reconstructed left and right states
        leftConsVars = primConsLib.primToCons_mhd(leftPrimVars, faceBx, gamma)
        rightConsVars = primConsLib.primToCons_mhd(rightPrimVars, faceBx, gamma)
        #Find the intercell fluxes between the left and right states
        intercellFlux = roeLib.riemannSolverX_mhd_roe(leftPrimVars, rightPrimVars, leftConsVars, rightConsVars,
                                                      faceBx, gamma, entropyFix, epsilon)
        leftIntercellFlux = intercellFlux[:,0:intercellFlux.shape[1]-1]
        rightIntercellFlux = intercellFlux[:,1:intercellFlux.shape[1]]
        #Update the conservative variables with Godunov's scheme
        dtdx = dt/dx
        consVarsX -= dtdx*(rightIntercellFlux - leftIntercellFlux)
        #Calculate the new primitive variable statues
        primVarsX[:,:] = primConsLib.consToPrim_mhd_x(consVarsX, Bx, gamma)
        #Update the full set of primitive variables
        primVarsAll = np.array([primVarsX[0],
                                primVarsX[1],
                                primVarsX[2],
                                primVarsX[3],
                                Bx,
                                primVarsX[4],
                                primVarsX[5],
                                primVarsX[6]])
        #Update the full set of conservative variables
        consVarsAll = np.array([consVarsX[0],
                                consVarsX[1],
                                consVarsX[2],
                                consVarsX[3],
                                Bx,
                                consVarsX[4],
                                consVarsX[5],
                                consVarsX[6]])
        #Update the time and cycle number
        time += dt
        cycle += 1
        #Print the time, cycle number, and timestep size
        print("Cycle = " + str(cycle) + " Time = " + str(time) + " Dt = " + str(dt))
    #Create a new primVarsAll matrix that can be returned by the evolve function
    #(i.e., put Bx back into the variable set)
    newPrimVarsAll = np.array([primVarsX[0],
                               primVarsX[1],
                               primVarsX[2],
                               primVarsX[3],
                               Bx,
                               primVarsX[4],
                               primVarsX[5],
                               primVarsX[6],])
    #Create a new consVarsAll matrix that can be returned by the evolve function
    #(i.e., put Bx back into the variable set)
    newConsVarsAll = np.array([consVarsX[0],
                               consVarsX[1],
                               consVarsX[2],
                               consVarsX[3],
                               Bx,
                               consVarsX[4],
                               consVarsX[5],
                               consVarsX[6],])
    #Return the final and second-last states of the primitive + conservative variables and the time
    #and cycle when we stopped the time evolution
    return newPrimVarsAll, newConsVarsAll, time, cycle, secondLastPrimVars, secondLastConsVars, secondLastTime



#####2D MHD EVOLVE FUNCTION#####

#Function: evolve_2D_mhd
#Purpose: Simulates the time evolution of a 2D magnetohydrodynamic simulation from
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
#                                    (0 = none, 2 = ppm), 0 by default)
#                  entropyFix (boolean flag for whether we should apply an entropy fix
#                              during the intercell flux calculations (see the
#                              RiemannSolvers_MHD_Roe_HelperModule.py script in Source
#                              if you want information on the purpose of entropy fixes
#                              and whether using one would be beneficial in your simulation).
#                  epsilon (epsilon value for the entropy fix, 0.5 by default)
#Outputs: primVars (the primitive variable states at time t = tEnd (or the primitive
#                   variable states at the moment when we reached maxCycles))
#         consVars (the conservative variable states at time t = tEnd (or the primitive
#                   variable states at the moment when we reached maxCycles))
#         newFaceBx (the new face-centred Bx values)
#         newFaceBy (the new face-centred By values)
#         time (the new time in the simulation (should equal tEnd if we didn't reach maxCycles))
#         cycle (the new cycle number for the simulation)
#         secondLastPrimVars (the state of the primitive variables at the end of the second-last cycle)
#         secondLastConsVars (the state of the conservative variables at the end of the second-last cycle)
#         secondLastFaceBx (the face-centred Bx values at the end of the second-last cycle)
#         secondLastFaceBy (the face-centred By values at the end of the second-last cycle)
#         secondLastTime (the time at the end of the second-last cycle)
def evolve_2D_mhd(simulationGrid, tStart, tEnd, startCycle, maxCycles, gamma, cfl, minDens, minPres, minEnergy,
                  reconstructOrder=0, entropyFix=False, epsilon=0.5):
    #Retrieve the primitive variables from the simulation grid
    primVarsAll = simulationGrid.primVars
    #Retrieve the boundary conditions for the edges of the simulation grid
    BcX = simulationGrid.BcX
    assert (BcX == constants.OUTFLOW or BcX == constants.PERIODIC)
    BcY = simulationGrid.BcY
    assert (BcY == constants.OUTFLOW or BcY == constants.PERIODIC)
    #Ensure that the simulation grid is 2D
    assert(len(primVarsAll.shape) == 3)
    #Ensure that 8 state variables have been defined for each cell
    assert(primVarsAll.shape[0] == 8)
    #Ensure that the number of cells is greater than zero
    assert(primVarsAll.shape[1] > 0)
    faceBx = simulationGrid.faceBx
    faceBy = simulationGrid.faceBy
    #Retrieve the conservative variables from the simulation grid
    consVarsAll = simulationGrid.consVars
    assert(primVarsAll.shape == consVarsAll.shape)
    #Create a primitive variables matrix for the x-direction
    #(i.e., include all primitive variables except Bx)
    primVarsX = np.array([primVarsAll[0],
                          primVarsAll[1],
                          primVarsAll[2],
                          primVarsAll[3],
                          primVarsAll[5],
                          primVarsAll[6],
                          primVarsAll[7]])
    #Create a primitive variables matrix for the y-direction
    #(i.e., include all primitive variables except By)
    primVarsY = np.array([primVarsAll[0],
                          primVarsAll[1],
                          primVarsAll[2],
                          primVarsAll[3],
                          primVarsAll[6],
                          primVarsAll[4],
                          primVarsAll[7]])
    #Create a conservative variables matrix for the x-direction
    consVarsX = np.array([consVarsAll[0],
                          consVarsAll[1],
                          consVarsAll[2],
                          consVarsAll[3],
                          consVarsAll[5],
                          consVarsAll[6],
                          consVarsAll[7]])
    #Retrieve the cell width from the simulation grid
    dx = simulationGrid.dx
    assert(dx > 0)
    #Retrieve the cell height from the simulation grid
    dy = simulationGrid.dy
    assert (dy > 0)

    #Create a variable for keeping track of the simulation time
    time = tStart
    assert(tStart <= tEnd)
    #Create a variable for keeping track of the cycle number
    cycle = startCycle
    assert(startCycle >= 0)

    #Initialize the second-last primitive variables
    secondLastPrimVars = np.copy(primVarsAll)
    #Initialize the second-last conservative variables
    secondLastConsVars = np.copy(consVarsAll)
    #Initialize the second-last face-centred Bx values
    secondLastFaceBx = np.copy(faceBx)
    #Initialize the second-last face-centred By values
    secondLastFaceBy = np.copy(faceBy)
    #Initialize the second-last time
    secondLastTime = tStart
    #Keep iterating until we reach tEnd or the max cycle number
    while time < tEnd and cycle < maxCycles:
        #Update the second-last variables and time
        secondLastPrimVars = np.copy(primVarsAll)
        secondLastConsVars = np.copy(consVarsAll)
        secondLastFaceBx = np.copy(faceBx)
        secondLastFaceBy = np.copy(faceBy)
        secondLastTime = time
        #Get the timestep size for this cycle
        dt = dtLib.calcDt_2D_mhd(consVarsAll,faceBx,faceBy,gamma,cfl,dx,dy)
        #Make sure we don't go over tEnd
        if time + dt > tEnd:
            dt = tEnd - time
        #Calculate some timestep-related values that will be helpful
        #for Godunov's scheme and the CTU+CT method
        dtdx = dt/dx
        dtdy = dt/dy
        halfDtDx = 0.5*dtdx
        halfDtDy = 0.5*dtdy
        #Start the cycle by performing spatial reconstruction
        #Check which spatial reconstruction order the user requested
        if reconstructOrder == 2: #if we should use PPM reconstruction
            (leftPrimVars, rightPrimVars) = ppmLib.ppmReconstructX_mhd(primVarsX, primVarsY[5], gamma, dt, dx, BcX)
            (topPrimVars, bottomPrimVars) = ppmLib.ppmReconstructY_mhd(primVarsY, primVarsX[4], gamma, dt, dy, BcY)
        # elif reconstructOrder == 1: #if we should use PLM reconstruction
        #     (leftPrimVars, rightPrimVars) = plmLib.ppmReconstruct1D_hydro(primVars, gamma, dt, dx, BcX)
        else: #if we shouldn't perform any spatial reconstruction
            #Apply boundary conditions to determine the correct left and right
            #variable states at the edges of the grid
            if BcX == 0: #if the BC is outflow
                leftPrimVars = np.append(primVarsX[:,:,0].reshape(primVarsX.shape[0],-1,1),
                                         primVarsX[:,:,:],axis=2)
                rightPrimVars = np.append(primVarsX[:,:,:],
                                          primVarsX[:,:,primVarsX.shape[2]-1].reshape(primVarsX.shape[0],-1,1),axis=2)
            else: #if the BC is periodic
                leftPrimVars = np.append(primVarsX[:,:,primVarsX.shape[2]-1].reshape(primVarsX.shape[0],-1,1),
                                         primVarsX[:,:,:],axis=2)
                rightPrimVars = np.append(primVarsX[:,:,:],
                                          primVarsX[:,:,0].reshape(primVarsX.shape[0],-1,1),axis=2)
            if BcY == 0: #if the BC is outflow
                topPrimVars = np.append(primVarsY[:,0,:].reshape(primVarsY.shape[0],1,-1),
                                        primVarsY[:,:,:],axis=1)
                bottomPrimVars = np.append(primVarsY[:,:,:],
                                           primVarsY[:,primVarsY.shape[1]-1,:].reshape(primVarsY.shape[0],1,-1),axis=1)
            else: #if the BC is periodic
                topPrimVars = np.append(primVarsY[:,primVarsY.shape[1]-1,:].reshape(primVarsY.shape[0],1,-1),
                                        primVarsY[:,:,:],axis=1)
                bottomPrimVars = np.append(primVarsY[:,:,:],
                                           primVarsY[:,0,:].reshape(primVarsY.shape[0],1,-1),axis=1)
        #Apply density and pressure floors to the reconstructed states
        leftPrimVars = primConsLib.applyFloors(leftPrimVars,True,minDens,minPres,minEnergy,
                                                "the x-direction reconstruction outputs (left)")
        rightPrimVars = primConsLib.applyFloors(rightPrimVars,True,minDens,minPres,minEnergy,
                                                "the x-direction reconstruction outputs (right)")
        topPrimVars = primConsLib.applyFloors(topPrimVars,True,minDens,minPres,minEnergy,
                                                "the y-direction reconstruction outputs (top)")
        bottomPrimVars = primConsLib.applyFloors(bottomPrimVars,True,minDens,minPres,minEnergy,
                                                "the y-direction reconstruction outputs (bottom)")

        #Add the MHD source terms to the reconstructed By values (see Reconstruction_PPM_MHD_HelperModule.py for
        #information on why these source terms are necessary)
        ByCorrections = ppmLib.getByReconSourceTerms_2D(primVarsX[2],faceBx,dt,dx,BcX)
        leftPrimVars[4] += ByCorrections[0]
        rightPrimVars[4] += ByCorrections[1]
        #Get the conservative form of the reconstructed left and right states
        leftConsVars = primConsLib.primToCons_mhd(leftPrimVars, faceBx, gamma)
        rightConsVars = primConsLib.primToCons_mhd(rightPrimVars, faceBx, gamma)

        #Add the MHD source terms to the reconstructed Bx values (see Reconstruction_PPM_MHD_HelperModule.py for
        #information on why these source terms are necessary)
        BxCorrections = ppmLib.getBxReconSourceTerms_2D(primVarsX[1], faceBy, dt, dy, BcY)
        topPrimVars[5] += BxCorrections[0]
        bottomPrimVars[5] += BxCorrections[1]
        #Get the conservative form of the reconstructed top and bottom states
        topConsVars = primConsLib.primToCons_mhd(topPrimVars, faceBy, gamma)
        bottomConsVars = primConsLib.primToCons_mhd(bottomPrimVars, faceBy, gamma)

        #Calculate the cell-centred electric field/EMF values as -v x B (= vy*Bx - vx*By)
        centEField = primVarsX[2]*primVarsY[5] - primVarsX[1]*primVarsX[4]

        #Find the intercell fluxes between the left and right states
        intercellFluxX = roeLib.riemannSolverX_mhd_roe(leftPrimVars, rightPrimVars, leftConsVars, rightConsVars, faceBx, gamma, entropyFix, epsilon)
        leftIntercellFlux = intercellFluxX[:,:,0:intercellFluxX.shape[2] - 1]
        rightIntercellFlux = intercellFluxX[:,:,1:intercellFluxX.shape[2]]

        #Find the intercell fluxes between the top and bottom states
        intercellFluxY = roeLib.riemannSolverY_mhd_roe(topPrimVars, bottomPrimVars, topConsVars, bottomConsVars, faceBy, gamma, entropyFix, epsilon)
        topIntercellFlux = intercellFluxY[:,0:intercellFluxY.shape[1]-1,:]
        bottomIntercellFlux = intercellFluxY[:,1:intercellFluxY.shape[1],:]

        #Calculate the corner EMF values
        cornerEMF = emfLib.getCornerEMF_2D(intercellFluxX, intercellFluxY, centEField, BcX, BcY)

        #Get the CTU half-timestep updates for the left and right reconstructed states
        horizontalUpdates = ctuLib.getXTransverseUpdates_2D(primVarsX, primVarsY, intercellFluxY,
                                                            faceBx, dt, dx, dy, BcX)
        leftConsVars += horizontalUpdates[0]
        rightConsVars += horizontalUpdates[1]
        #Apply density and pressure floors to the updated reconstructed states
        leftConsVars = primConsLib.applyFloors(leftConsVars,False,minDens,minPres,minEnergy,
                                                "the x-direction reconstruction outputs (left) after CTU update")
        rightConsVars = primConsLib.applyFloors(rightConsVars,False,minDens,minPres,minEnergy,
                                                "the x-direction reconstruction outputs (right) after CTU update")

        #Get the CTU half-timestep updates for the top and bottom reconstructed states
        verticalUpdates = ctuLib.getYTransverseUpdates_2D(primVarsX, primVarsY, intercellFluxX,
                                                          faceBy, dt, dx, dy, BcY)
        topConsVars += verticalUpdates[0]
        bottomConsVars += verticalUpdates[1]
        #Apply density and pressure floors to the updated reconstructed states
        topConsVars = primConsLib.applyFloors(topConsVars,False,minDens,minPres,minEnergy,
                                              "the y-direction reconstruction outputs (top) after CTU update")
        bottomConsVars = primConsLib.applyFloors(bottomConsVars,False,minDens,minPres,minEnergy,
                                                "the y-direction reconstruction outputs (bottom) after CTU update")

        #Use the EMF values to evolve the face-centred Bx values by half a timestep
        lowerCornerEMF = cornerEMF[1:cornerEMF.shape[0],:]
        upperCornerEMF = cornerEMF[0:cornerEMF.shape[0]-1,:]
        newFaceBx = faceBx - halfDtDy*(lowerCornerEMF-upperCornerEMF)

        #Use the EMF values to evolve the face-centred By values by half a timestep
        rightCornerEMF = cornerEMF[:,1:cornerEMF.shape[1]]
        leftCornerEMF = cornerEMF[:,0:cornerEMF.shape[1]-1]
        newFaceBy = faceBy + halfDtDx*(rightCornerEMF-leftCornerEMF)

        #Calculate the primitive form of the CTU-updated reconstructed conservative variable states
        leftPrimVars = primConsLib.consToPrim_mhd(leftConsVars,newFaceBx,gamma)
        rightPrimVars = primConsLib.consToPrim_mhd(rightConsVars,newFaceBx,gamma)
        topPrimVars = primConsLib.consToPrim_mhd(topConsVars,newFaceBy,gamma)
        bottomPrimVars = primConsLib.consToPrim_mhd(bottomConsVars,newFaceBy,gamma)

        #Now we will use Godunov's scheme to update the cell-centred conservative variables (except Bx and By)
        #by half a timestep.

        #First, we need to make some changes to our intercell flux vectors.

        #In the y-direction, the Bz flux is the second magnetic field component, whereas it is the first
        #in the x-direction. We, therefore, move the vertical Bz fluxes to the same index as the Bz component in
        #the x-direction flux vector.
        topIntercellFlux[5] = topIntercellFlux[4]
        bottomIntercellFlux[5] = bottomIntercellFlux[4]
        #We will now set the Bx fluxes to zero in the y-direction and the By fluxes to zero in the x-direction
        #(because we use the EMF formulas to update the in-plane magnetic field components instead of Godunov's scheme).
        topIntercellFlux[4] = 0.0
        bottomIntercellFlux[4] = 0.0
        leftIntercellFlux[4] = 0.0
        rightIntercellFlux[4] = 0.0

        #Use Godunov's scheme to update the cell-centred conservative variables (except Bx and By)
        newConsVarsX = consVarsX - halfDtDx*(rightIntercellFlux - leftIntercellFlux)
        newConsVarsX -= halfDtDy*(bottomIntercellFlux - topIntercellFlux)

        #Get the new cell-centred Bx and By values from the half-timestep face-centred magnetic fields
        newCentBx = ctLib.getCentBx(newFaceBx)
        newCentBy = ctLib.getCentBy(newFaceBy)

        #Get the new cell-centred x- and y-velocities
        newVx = newConsVarsX[1]/newConsVarsX[0]
        newVy = newConsVarsX[2]/newConsVarsX[0]

        #Calculate the new cell-centred electric field/EMF values
        newCentEField = newVy*newCentBx - newVx*newCentBy

        #Find the intercell fluxes between the left and right states (at time t + dt/2)
        newIntercellFluxX = roeLib.riemannSolverX_mhd_roe(leftPrimVars, rightPrimVars, leftConsVars, rightConsVars, newFaceBx, gamma, entropyFix, epsilon)
        leftIntercellFlux = newIntercellFluxX[:,:,0:intercellFluxX.shape[2]-1]
        rightIntercellFlux = newIntercellFluxX[:,:,1:intercellFluxX.shape[2]]

        #Find the intercell fluxes between the top and bottom states (at time + dt/2)
        newIntercellFluxY = roeLib.riemannSolverY_mhd_roe(topPrimVars, bottomPrimVars, topConsVars, bottomConsVars, newFaceBy, gamma, entropyFix, epsilon)
        topIntercellFlux = newIntercellFluxY[:,0:intercellFluxY.shape[1]-1,:]
        bottomIntercellFlux = newIntercellFluxY[:,1:intercellFluxY.shape[1],:]

        #Calculate the corner EMF values (at time t + dt/2)
        cornerEMF = emfLib.getCornerEMF_2D(newIntercellFluxX, newIntercellFluxY, newCentEField, BcX, BcY)
        lowerCornerEMF = cornerEMF[1:cornerEMF.shape[0],:]
        upperCornerEMF = cornerEMF[0:cornerEMF.shape[0]-1,:]
        rightCornerEMF = cornerEMF[:,1:cornerEMF.shape[1]]
        leftCornerEMF = cornerEMF[:,0:cornerEMF.shape[1]-1]

        #Evolve the face-centred magnetic fields by a full timestep
        faceBx -= (dt/dy)*(lowerCornerEMF-upperCornerEMF)
        faceBy += (dt/dx)*(rightCornerEMF-leftCornerEMF)

        #Get the new cell-centred Bx and By values
        centBx = ctLib.getCentBx(faceBx)
        centBy = ctLib.getCentBy(faceBy)

        #Now we will use Godunov's scheme to calculate the rest of the conservative variables at time t + dt

        #Calculate the new Bz values
        oldBz = consVarsX[5]
        newBz = oldBz - (dt/dx)*(rightIntercellFlux[5]-leftIntercellFlux[5]) \
                      - (dt/dy)*(bottomIntercellFlux[4]-topIntercellFlux[4])
        #Update the rest of the variables in consVarsX (except By)
        consVarsX = consVarsX - (dt/dx)*(rightIntercellFlux-leftIntercellFlux)
        consVarsX -= (dt / dy) * (bottomIntercellFlux - topIntercellFlux)
        #Put the new Bz and By values into consVarsX
        consVarsX[5] = newBz
        consVarsX[4] = centBy

        #Get the new primitive variable states in the x-direction
        primVarsX = primConsLib.consToPrim_mhd(consVarsX,centBx,gamma)
        #Update the primitive variable states in the y-direction
        primVarsY = np.copy(primVarsX)
        primVarsY[4] = newBz
        primVarsY[5] = centBx
        #Update the full set of primitive variables
        primVarsAll = np.array([primVarsX[0],
                                primVarsX[1],
                                primVarsX[2],
                                primVarsX[3],
                                centBx,
                                primVarsX[4],
                                primVarsX[5],
                                primVarsX[6]])
        #Update the full set of conservative variables
        consVarsAll = np.array([consVarsX[0],
                                consVarsX[1],
                                consVarsX[2],
                                consVarsX[3],
                                centBx,
                                consVarsX[4],
                                consVarsX[5],
                                consVarsX[6]])
        #Update the time and cycle number
        time += dt
        cycle += 1
        #Print the time, cycle number, and timestep size
        print("Cycle = " + str(cycle) + " Time = " + str(time) + " Dt = " + str(dt))

    #Return the final state of the primitive + conservative variables and the time
    #and cycle when we stopped the time evolution
    return primVarsAll, consVarsAll, faceBx, faceBy, time, cycle, \
           secondLastPrimVars, secondLastConsVars, secondLastFaceBx, secondLastFaceBy, secondLastTime

#####3D MHD EVOLVE FUNCTION#####

#Function: evolve_3D_mhd
#Purpose: Simulates the time evolution of a 3D magnetohydrodynamic simulation from
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
#                                    (0 = none, 2 = ppm), 0 by default)
#                  entropyFix (boolean flag for whether we should apply an entropy fix
#                              during the intercell flux calculations (see the
#                              RiemannSolvers_MHD_Roe_HelperModule.py script in Source
#                              if you want information on the purpose of entropy fixes
#                              and whether using one would be beneficial in your simulation).
#                  epsilon (epsilon value for the entropy fix, 0.5 by default)
#Outputs: primVars (the primitive variable states at time t = tEnd (or the primitive
#                   variable states at the moment when we reached maxCycles))
#         consVars (the conservative variable states at time t = tEnd (or the primitive
#                   variable states at the moment when we reached maxCycles))
#         time (the new time in the simulation (should equal tEnd if we didn't reach maxCycles))
#         cycle (the new cycle number for the simulation)
#         newFaceBx (the new face-centred Bx values)
#         newFaceBy (the new face-centred By values)
#         newFaceBz (the new face-centred Bz values)
#         secondLastPrimVars (the state of the primitive variables at the end of the second-last cycle)
#         secondLastConsVars (the state of the conservative variables at the end of the second-last cycle)
#         secondLastTime (the time at the end of the second-last cycle)
#         secondLastFaceBx (the face-centred Bx values at the end of the second-last cycle)
#         secondLastFaceBy (the face-centred By values at the end of the second-last cycle)
#         secondLastFaceBz (the face-centred Bz values at the end of the second-last cycle)
def evolve_3D_mhd(simulationGrid, tStart, tEnd, startCycle, maxCycles, gamma, cfl, minDens, minPres, minEnergy,
                  reconstructOrder=0, entropyFix=True, epsilon=0.5):
    #Retrieve the primitive variables from the simulation grid
    primVarsAll = simulationGrid.primVars
    # Retrieve the boundary conditions for the edges of the simulation grid
    BcX = simulationGrid.BcX
    assert (BcX == constants.OUTFLOW or BcX == constants.PERIODIC)
    BcY = simulationGrid.BcY
    assert (BcY == constants.OUTFLOW or BcY == constants.PERIODIC)
    BcZ = simulationGrid.BcZ
    assert (BcZ == constants.OUTFLOW or BcZ == constants.PERIODIC)
    #Ensure that the simulation grid is 3D
    assert(len(primVarsAll.shape) == 4)
    #Ensure that 8 state variables have been defined for each cell
    assert(primVarsAll.shape[0] == 8)
    #Ensure that the number of cells is greater than zero
    assert(primVarsAll.shape[1] > 0)
    faceBx = simulationGrid.faceBx
    faceBy = simulationGrid.faceBy
    faceBz = simulationGrid.faceBz
    #Retrieve the conservative variables from the simulation grid
    consVarsAll = simulationGrid.consVars
    assert(primVarsAll.shape == consVarsAll.shape)
    #Create a primitive variables matrix for the x-direction
    #(i.e., include all primitive variables except Bx)
    primVarsX = np.array([primVarsAll[0],
                          primVarsAll[1],
                          primVarsAll[2],
                          primVarsAll[3],
                          primVarsAll[5],
                          primVarsAll[6],
                          primVarsAll[7]])
    #Create a primitive variables matrix for the y-direction
    #(i.e., include all primitive variables except By)
    primVarsY = np.array([primVarsAll[0],
                          primVarsAll[1],
                          primVarsAll[2],
                          primVarsAll[3],
                          primVarsAll[6],
                          primVarsAll[4],
                          primVarsAll[7]])
    # Create a primitive variables matrix for the z-direction
    # (i.e., include all primitive variables except Bz)
    primVarsZ = np.array([primVarsAll[0],
                          primVarsAll[1],
                          primVarsAll[2],
                          primVarsAll[3],
                          primVarsAll[4],
                          primVarsAll[5],
                          primVarsAll[7]])
    #Create a conservative variables matrix for the x-direction
    consVarsX = np.array([consVarsAll[0],
                          consVarsAll[1],
                          consVarsAll[2],
                          consVarsAll[3],
                          consVarsAll[5],
                          consVarsAll[6],
                          consVarsAll[7]])

    #Retrieve the cell width from the simulation grid
    dx = simulationGrid.dx
    assert(dx > 0)
    # Retrieve the cell height from the simulation grid
    dy = simulationGrid.dy
    assert (dy > 0)
    #Retrieve the cell depth from the simulation grid
    dz = simulationGrid.dz
    assert (dy > 0)

    #Create a variable for keeping track of the simulation time
    time = tStart
    assert(tStart <= tEnd)
    #Create a variable for keeping track of the cycle number
    cycle = startCycle
    assert(startCycle >= 0)

    #Initialize the second-last primitive variables
    secondLastPrimVars = np.copy(primVarsAll)
    #Initialize the second-last conservative variables
    secondLastConsVars = np.copy(consVarsAll)
    #Initialize the second-last face-centred Bx values
    secondLastFaceBx = np.copy(faceBx)
    #Initialize the second-last face-centred By values
    secondLastFaceBy = np.copy(faceBy)
    #Initialize the second-last face-centred Bz values
    secondLastFaceBz = np.copy(faceBz)
    #Initialize the second-last time
    secondLastTime = tStart

    #Keep iterating until we reach tEnd or the max cycle number
    while time < tEnd and cycle < maxCycles:
        #Update the second-last variables and time
        secondLastPrimVars = np.copy(primVarsAll)
        secondLastConsVars = np.copy(consVarsAll)
        secondLastFaceBx = np.copy(faceBx)
        secondLastFaceBy = np.copy(faceBy)
        secondLastFaceBz = np.copy(faceBz)
        secondLastTime = time
        #Get the timestep size for this cycle
        dt = dtLib.calcDt_3D_mhd(consVarsAll,faceBx,faceBy,faceBz,gamma,cfl,dx,dy,dz)
        #Make sure we don't go over tEnd
        if tEnd - time < dt:
            dt = tEnd - time
        #Calculate some timestep-related values that will be helpful
        #for Godunov's scheme and the CTU+CT method
        dtdx = dt/dx
        dtdy = dt/dy
        dtdz = dt/dz
        halfDtDx = 0.5*dtdx
        halfDtDy = 0.5*dtdy
        halfDtDz = 0.5*dtdz
        #Start the cycle by performing spatial reconstruction
        #Check which spatial reconstruction order the user requested
        if reconstructOrder == 2: #if we should use PPM reconstruction
            (leftPrimVars, rightPrimVars) = ppmLib.ppmReconstructX_mhd(primVarsX, primVarsY[5], gamma, dt, dx, BcX)
            (topPrimVars, bottomPrimVars) = ppmLib.ppmReconstructY_mhd(primVarsY, primVarsX[4], gamma, dt, dy, BcY)
            (backPrimVars, forwPrimVars) = ppmLib.ppmReconstructZ_mhd(primVarsZ, primVarsX[5], gamma, dt, dz, BcZ)
        # elif reconstructOrder == 1: #if we should use PLM reconstruction
        #     (leftPrimVars, rightPrimVars) = plmLib.ppmReconstruct1D_hydro(primVars, gamma, dt, dx, BcX)
        else: #if we shouldn't perform any spatial reconstruction
            #Apply boundary conditions to determine the correct left and right
            #variable states at the edges of the grid
            if BcX == 0: #if the BC is outflow
                leftPrimVars = np.append(primVarsX[:,:,0,:].reshape(primVarsX.shape[0],primVarsX.shape[1],1,-1),
                                         primVarsX[:,:,:,:],axis=2)
                rightPrimVars = np.append(primVarsX[:,:,:,:],
                                          primVarsX[:,:,primVarsX.shape[2]-1,:].reshape(primVarsX.shape[0],
                                                                                        primVarsX.shape[1],1,-1),axis=2)
            else: #if the BC is periodic
                leftPrimVars = np.append(primVarsX[:,:,primVarsX.shape[2]-1,:].reshape(primVarsX.shape[0],
                                                                                       primVarsX.shape[1],1,-1),
                                         primVarsX[:,:,:,:],axis=2)
                rightPrimVars = np.append(primVarsX[:,:,:,:],
                                          primVarsX[:,:,0,:].reshape(primVarsX.shape[0],primVarsX.shape[1],1,-1),axis=2)
            if BcY == 0: #if the BC is outflow
                topPrimVars = np.append(primVarsY[:,0,:,:].reshape(primVarsY.shape[0],1,-1),
                                        primVarsY[:,:,:,:],axis=1)
                bottomPrimVars = np.append(primVarsY[:,:,:,:],
                                           primVarsY[:,primVarsY.shape[1]-1,:,:].reshape(primVarsY.shape[0],1,
                                                                                         primVarsY.shape[2],-1),axis=1)
            else: #if the BC is periodic
                topPrimVars = np.append(primVarsY[:,primVarsY.shape[1]-1,:,:].reshape(primVarsY.shape[0],1,
                                                                                      primVarsY.shape[2],-1),
                                        primVarsY[:,:,:,:],axis=1)
                bottomPrimVars = np.append(primVarsY[:,:,:,:],
                                           primVarsY[:,0,:,:].reshape(primVarsY.shape[0],1,primVarsY.shape[2],-1),axis=1)
            #Apply boundary conditions to determine the correct back and forward
            #variable states at the edges of the grid
            if BcX == 0: #if the BC is outflow
                backPrimVars = np.append(primVarsZ[:,:,:,0].reshape(primVarsZ.shape[0],primVarsZ.shape[1],-1,1),
                                         primVarsZ[:,:,:,:],axis=3)
                forwPrimVars = np.append(primVarsZ[:,:,:,:],
                                         primVarsZ[:,:,:,primVarsZ.shape[3]-1].reshape(primVarsZ.shape[0],
                                                                                       primVarsZ.shape[1],-1,1),axis=3)
            else: #if the BC is periodic
                backPrimVars = np.append(primVarsZ[:,:,:,primVarsZ.shape[3]-1].reshape(primVarsZ.shape[0],
                                                                                       primVarsZ.shape[1],-1,1),
                                         primVarsZ[:,:,:,:],axis=3)
                forwPrimVars = np.append(primVarsZ[:,:,:,:],
                                          primVarsZ[:,:,:,0].reshape(primVarsZ.shape[0],primVarsZ.shape[1],-1,1),axis=3)
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
                                                "the z-direction reconstruction outputs (forw)")

        #Add the MHD source terms to the reconstructed By and Bz values in the x-direction
        #(see Reconstruction_PPM_MHD_HelperModule.py for information on why these source terms are necessary)
        xBCorrections = ppmLib.getReconBCorrectionsX_3D(consVarsX[2],consVarsX[3],primVarsX[0],
                                                        faceBx,faceBy,faceBz,dt,dx,dy,dz,BcX)
        leftPrimVars[4] += xBCorrections[0,0]
        leftPrimVars[5] += xBCorrections[0,1]
        rightPrimVars[4] += xBCorrections[1,0]
        rightPrimVars[5] += xBCorrections[1,1]
        #Get the conservative form of the reconstructed primitive variables
        leftConsVars = primConsLib.primToCons_mhd_x(leftPrimVars, faceBx, gamma)
        rightConsVars = primConsLib.primToCons_mhd_x(rightPrimVars, faceBx, gamma)

        #Add the MHD source terms to the reconstructed Bz and Bx values in the y-direction
        #(see Reconstruction_PPM_MHD_HelperModule.py for information on why these source terms are necessary)
        yBCorrections = ppmLib.getReconBCorrectionsY_3D(consVarsX[1],consVarsX[3],primVarsX[0],
                                                        faceBx,faceBy,faceBz,dt,dx,dy,dz,BcY)
        topPrimVars[4] += yBCorrections[0,0]
        topPrimVars[5] += yBCorrections[0,1]
        bottomPrimVars[4] += yBCorrections[1,0]
        bottomPrimVars[5] += yBCorrections[1,1]
        #Get the conservative form of the reconstructed primitive variables
        topConsVars = primConsLib.primToCons_mhd_y(topPrimVars, faceBy, gamma)
        bottomConsVars = primConsLib.primToCons_mhd_y(bottomPrimVars, faceBy, gamma)

        #Add the MHD source terms to the reconstructed Bx and Bx values in the z-direction
        #(see Reconstruction_PPM_MHD_HelperModule.py for information on why these source terms are necessary)
        zBCorrections = ppmLib.getReconBCorrectionsZ_3D(consVarsX[1],consVarsX[2],primVarsX[0],
                                                        faceBx,faceBy,faceBz,dt,dx,dy,dz,BcX)
        backPrimVars[4] += zBCorrections[0,0]
        backPrimVars[5] += zBCorrections[0,1]
        forwPrimVars[4] += zBCorrections[1,0]
        forwPrimVars[5] += zBCorrections[1,1]
        #Get the conservative form of the reconstructed primitive variables
        backConsVars = primConsLib.primToCons_mhd_z(backPrimVars, faceBz, gamma)
        forwConsVars = primConsLib.primToCons_mhd_z(forwPrimVars, faceBz, gamma)

        #Calculate the cell-centred electric fields
        centEx = (primVarsX[4]*consVarsX[3] - primVarsX[5]*consVarsX[2])/consVarsX[0]
        centEy = (primVarsX[5]*consVarsX[1] - primVarsY[5]*consVarsX[3])/consVarsX[0]
        centEz = (primVarsY[5]*consVarsX[2] - primVarsX[4]*consVarsX[1])/consVarsX[0]

        #Find the intercell fluxes between the left and right states
        intercellFluxX = roeLib.riemannSolverX_mhd_roe(leftPrimVars, rightPrimVars, leftConsVars, rightConsVars, faceBx, gamma, entropyFix, epsilon)
        leftIntercellFlux = intercellFluxX[:,:,0:intercellFluxX.shape[2] - 1]
        rightIntercellFlux = intercellFluxX[:,:,1:intercellFluxX.shape[2]]

        #Find the intercell fluxes between the top and bottom states
        intercellFluxY = roeLib.riemannSolverY_mhd_roe(topPrimVars, bottomPrimVars, topConsVars, bottomConsVars, faceBy, gamma, entropyFix, epsilon)
        topIntercellFlux = intercellFluxY[:,0:intercellFluxY.shape[1]-1,:]
        bottomIntercellFlux = intercellFluxY[:,1:intercellFluxY.shape[1],:]

        #Find the intercell fluxes between the back and forward states
        intercellFluxZ = roeLib.riemannSolverZ_mhd_roe(backPrimVars, forwPrimVars, backConsVars, forwConsVars, faceBz, gamma, entropyFix, epsilon)
        backIntercellFlux = intercellFluxZ[:,:,:,0:intercellFluxZ.shape[3]-1]
        forwIntercellFlux = intercellFluxZ[:,:,:,1:intercellFluxZ.shape[3]]

        #Get the corner EMF values that we will use to evolve the face-centred magnetic fields
        cornerEMF_x = emfLib.getXCornerEMF_3D(intercellFluxY, intercellFluxZ, centEx, BcX, BcY, BcZ)
        cornerEMF_y = emfLib.getYCornerEMF_3D(intercellFluxX, intercellFluxZ, centEy, BcX, BcY, BcZ)
        cornerEMF_z = emfLib.getZCornerEMF_3D(intercellFluxX, intercellFluxY, centEz, BcX, BcY, BcZ)

        #Evolve the left and right reconstructed conservative states by half a timestep
        (leftConsVars,rightConsVars) = ctuLib.getXTransverseUpdates_3D(leftConsVars,rightConsVars,primVarsX,primVarsY,intercellFluxY,intercellFluxZ,faceBx,
                                                            faceBy,faceBz,cornerEMF_x,dt,dx,dy,dz,BcX)
        # leftConsVars += horizontalUpdates[0]
        # rightConsVars += horizontalUpdates[1]

        #Evolve the top and bottom reconstructed conservative states by half a timestep
        (topConsVars, bottomConsVars) = ctuLib.getYTransverseUpdates_3D(topConsVars,bottomConsVars,primVarsX,primVarsY,intercellFluxX,intercellFluxZ,
                                                          faceBx,faceBy,faceBz,cornerEMF_y,dt,dx,dy,dz,BcY)
        # topConsVars += verticalUpdates[0]
        # bottomConsVars += verticalUpdates[1]

        #Evolve the back and forward reconstructed conservative states by half a timestep
        (backConsVars, forwConsVars) = ctuLib.getZTransverseUpdates_3D(backConsVars,forwConsVars,primVarsX,primVarsY,intercellFluxX,intercellFluxY,
                                                          faceBx,faceBy,faceBz,cornerEMF_z,dt,dx,dy,dz,BcZ)
        # backConsVars += depthUpdates[0]
        # forwConsVars += depthUpdates[1]

        #Evolve the face-centred Bx values by half a timestep
        lowerCornerEMF_z = cornerEMF_z[1:cornerEMF_z.shape[0],:,0:cornerEMF_z.shape[2]-1]
        upperCornerEMF_z = cornerEMF_z[0:cornerEMF_z.shape[0] - 1, :,0:cornerEMF_z.shape[2]-1]
        forwCornerEMF_y = cornerEMF_y[0:cornerEMF_y.shape[0] - 1, :, 1:cornerEMF_y.shape[2]]
        backCornerEMF_y = cornerEMF_y[0:cornerEMF_y.shape[0] - 1, :, 0:cornerEMF_y.shape[2] - 1]
        newFaceBx = np.copy(faceBx)
        newFaceBx += halfDtDz*(forwCornerEMF_y - backCornerEMF_y) \
                    - halfDtDy*(lowerCornerEMF_z - upperCornerEMF_z)

        #Evolve the face-centred By values by half a timestep
        rightCornerEMF_z = cornerEMF_z[:,1:cornerEMF_z.shape[1],0:cornerEMF_z.shape[2]-1]
        leftCornerEMF_z = cornerEMF_z[:,0:cornerEMF_z.shape[1]-1,0:cornerEMF_z.shape[2]-1]
        forwCornerEMF_x = cornerEMF_x[:,0:cornerEMF_x.shape[1]-1,1:cornerEMF_x.shape[2]]
        backCornerEMF_x = cornerEMF_x[:,0:cornerEMF_x.shape[1]-1,0:cornerEMF_x.shape[2]-1]
        newFaceBy = np.copy(faceBy)
        newFaceBy += halfDtDx*(rightCornerEMF_z - leftCornerEMF_z) \
                     - halfDtDz*(forwCornerEMF_x - backCornerEMF_x)

        #Evolve the face-centred Bz values by half a timestep
        lowerCornerEMF_x = cornerEMF_x[1:cornerEMF_x.shape[0],0:cornerEMF_x.shape[1]-1,:]
        upperCornerEMF_x = cornerEMF_x[0:cornerEMF_x.shape[0]-1,0:cornerEMF_x.shape[1]-1,:]
        rightCornerEMF_y = cornerEMF_y[0:cornerEMF_y.shape[0]-1,1:cornerEMF_y.shape[1],:]
        leftCornerEMF_y = cornerEMF_y[0:cornerEMF_y.shape[0]-1,0:cornerEMF_y.shape[1]-1,:]
        newFaceBz = np.copy(faceBz)
        newFaceBz += halfDtDy*(lowerCornerEMF_x - upperCornerEMF_x) \
                     - halfDtDx*(rightCornerEMF_y - leftCornerEMF_y)

        #Get the primitive forms of the t + dt/2 reconstructed states
        leftPrimVars = primConsLib.consToPrim_mhd_x(leftConsVars,newFaceBx,gamma)
        rightPrimVars = primConsLib.consToPrim_mhd_x(rightConsVars,newFaceBx,gamma)
        topPrimVars = primConsLib.consToPrim_mhd_y(topConsVars,newFaceBy,gamma)
        bottomPrimVars = primConsLib.consToPrim_mhd_y(bottomConsVars,newFaceBy,gamma)
        backPrimVars = primConsLib.consToPrim_mhd_z(backConsVars,newFaceBz, gamma)
        forwPrimVars = primConsLib.consToPrim_mhd_z(forwConsVars,newFaceBz, gamma)

        #Use Godunov's scheme to evolve the cell-centred density and momenta by half a timestep
        newDens = consVarsX[0] - halfDtDx*(rightIntercellFlux[0] - leftIntercellFlux[0]) \
                               - halfDtDy*(bottomIntercellFlux[0] - topIntercellFlux[0]) \
                               - halfDtDz*(forwIntercellFlux[0] - backIntercellFlux[0])
        newMomX = consVarsX[1] - halfDtDx*(rightIntercellFlux[1] - leftIntercellFlux[1]) \
                               - halfDtDy*(bottomIntercellFlux[1] - topIntercellFlux[1]) \
                               - halfDtDz*(forwIntercellFlux[1] - backIntercellFlux[1])
        newMomY = consVarsX[2] - halfDtDx*(rightIntercellFlux[2] - leftIntercellFlux[2]) \
                               - halfDtDy*(bottomIntercellFlux[2] - topIntercellFlux[2]) \
                               - halfDtDz*(forwIntercellFlux[2] - backIntercellFlux[2])
        newMomZ = consVarsX[3] - halfDtDx*(rightIntercellFlux[3] - leftIntercellFlux[3]) \
                               - halfDtDy*(bottomIntercellFlux[3] - topIntercellFlux[3]) \
                               - halfDtDz*(forwIntercellFlux[3] - backIntercellFlux[3])
        #Use t + dt/2 face-centred magnetic field components to calculate
        #the t + dt/2 cell-centred magnetic field components
        #(via the CT equations that are discussed at the top of this script)
        newBx = 0.5*(newFaceBx[:,0:newFaceBx.shape[1]-1,:] + newFaceBx[:,1:newFaceBx.shape[1],:])
        newBy = 0.5*(newFaceBy[0:newFaceBy.shape[0]-1,:,:] + newFaceBy[1:newFaceBy.shape[0],:,:])
        newBz = 0.5*(newFaceBz[:,:,0:newFaceBz.shape[2]-1] + newFaceBz[:,:,1:newFaceBz.shape[2]])
        #Calculate the t + dt/2 cell-centred electric field components
        newCentEx = (newBy*newMomZ - newBz*newMomY)/newDens
        newCentEy = (newBz*newMomX - newBx*newMomZ)/newDens
        newCentEz = (newBx*newMomY - newBy*newMomX)/newDens

        #Find the intercell fluxes between the left and right t + dt/2 states
        newIntercellFluxX = roeLib.riemannSolverX_mhd_roe(leftPrimVars, rightPrimVars, leftConsVars, rightConsVars, newFaceBx, gamma, entropyFix, epsilon)
        leftIntercellFlux = newIntercellFluxX[:,:,0:intercellFluxX.shape[2]-1]
        rightIntercellFlux = newIntercellFluxX[:,:,1:intercellFluxX.shape[2]]

        #Find the intercell fluxes between the top and bottom t + dt/2 states
        newIntercellFluxY = roeLib.riemannSolverY_mhd_roe(topPrimVars, bottomPrimVars, topConsVars, bottomConsVars, newFaceBy, gamma, entropyFix, epsilon)
        topIntercellFlux = newIntercellFluxY[:,0:intercellFluxY.shape[1]-1,:]
        bottomIntercellFlux = newIntercellFluxY[:,1:intercellFluxY.shape[1],:]

        #Find the intercell fluxes between the back and forward t + dt/2 states
        newIntercellFluxZ = roeLib.riemannSolverZ_mhd_roe(backPrimVars, forwPrimVars,  backConsVars, forwConsVars, newFaceBz, gamma, entropyFix, epsilon)
        backIntercellFlux = newIntercellFluxZ[:,:,:,0:intercellFluxZ.shape[3]-1]
        forwIntercellFlux = newIntercellFluxZ[:,:,:,1:intercellFluxZ.shape[3]]

        #Calculate the t + dt/2 corner EMF values
        newCornerEMF_x = emfLib.getXCornerEMF_3D(newIntercellFluxY, newIntercellFluxZ, newCentEx, BcX, BcY, BcZ)
        newCornerEMF_y = emfLib.getYCornerEMF_3D(newIntercellFluxX, newIntercellFluxZ, newCentEy, BcX, BcY, BcZ)
        newCornerEMF_z = emfLib.getZCornerEMF_3D(newIntercellFluxX, newIntercellFluxY, newCentEz, BcX, BcY, BcZ)

        #Using the t + dt/2 corner EMFs, evolve the face-centred Bx values by a full timestep
        lowerCornerEMF_z = newCornerEMF_z[1:cornerEMF_z.shape[0],:,0:cornerEMF_z.shape[2]-1]
        upperCornerEMF_z = newCornerEMF_z[0:cornerEMF_z.shape[0]-1,:,0:cornerEMF_z.shape[2]-1]
        forwCornerEMF_y = newCornerEMF_y[0:cornerEMF_y.shape[0]-1,:,1:cornerEMF_y.shape[2]]
        backCornerEMF_y = newCornerEMF_y[0:cornerEMF_y.shape[0]-1,:,0:cornerEMF_y.shape[2]-1]
        faceBx += dtdz*(forwCornerEMF_y - backCornerEMF_y)\
                  -dtdy*(lowerCornerEMF_z - upperCornerEMF_z)

        #Using the t + dt/2 corner EMFs, evolve the face-centred By values by a full timestep
        rightCornerEMF_z = newCornerEMF_z[:,1:cornerEMF_z.shape[1],0:cornerEMF_z.shape[2]-1]
        leftCornerEMF_z = newCornerEMF_z[:,0:cornerEMF_z.shape[1]-1,0:cornerEMF_z.shape[2]-1]
        forwCornerEMF_x = newCornerEMF_x[:,0:cornerEMF_x.shape[1]-1,1:cornerEMF_x.shape[2]]
        backCornerEMF_x = newCornerEMF_x[:,0:cornerEMF_x.shape[1]-1,0:cornerEMF_x.shape[2]-1]
        faceBy += dtdx*(rightCornerEMF_z - leftCornerEMF_z) \
                  -dtdz*(forwCornerEMF_x - backCornerEMF_x)

        #Using the t + dt/2 corner EMFs, evolve the face-centred Bz values by a full timestep
        lowerCornerEMF_x = newCornerEMF_x[1:cornerEMF_x.shape[0],0:cornerEMF_x.shape[1]-1,:]
        upperCornerEMF_x = newCornerEMF_x[0:cornerEMF_x.shape[0]-1,0:cornerEMF_x.shape[1]-1,:]
        rightCornerEMF_y = newCornerEMF_y[0:cornerEMF_y.shape[0]-1,1:cornerEMF_y.shape[1],:]
        leftCornerEMF_y = newCornerEMF_y[0:cornerEMF_y.shape[0]-1,0:cornerEMF_y.shape[1]-1,:]
        faceBz += dtdy*(lowerCornerEMF_x - upperCornerEMF_x) \
                   - dtdx*(rightCornerEMF_y - leftCornerEMF_y)

        #Evolve the cell-centred density, momenta, and energy by a full timestep
        consVarsX -= dtdx*(rightIntercellFlux - leftIntercellFlux)
        consVarsX -= dtdy*(bottomIntercellFlux - topIntercellFlux)
        consVarsX -= dtdz*(forwIntercellFlux - backIntercellFlux)
        #Calculate the new cell-centred magnetic field components
        centBx = 0.5*(faceBx[:,0:newFaceBx.shape[1]-1,:] + faceBx[:,1:newFaceBx.shape[1],:])
        centBy = 0.5*(faceBy[0:newFaceBy.shape[0]-1,:,:] + faceBy[1:newFaceBy.shape[0],:,:])
        centBz = 0.5*(faceBz[:,:,0:newFaceBz.shape[2]-1] + faceBz[:,:,1:newFaceBz.shape[2]])
        #Put the new By and Bz values into the x-direction conservative variables matrix
        consVarsX[4] = centBy
        consVarsX[5] = centBz

        #Get the new primitive variables in the x-direction
        primVarsX = primConsLib.consToPrim_mhd(consVarsX,centBx,gamma)

        #Update the y- and z-direction primitive variable matrices
        primVarsY = np.copy(primVarsX)
        primVarsY[4] = centBz
        primVarsY[5] = centBx
        primVarsZ = np.copy(primVarsX)
        primVarsZ[4] = centBx
        primVarsZ[5] = centBy

        #Update the full set of primitive variables
        primVarsAll = np.array([primVarsX[0],
                                primVarsX[1],
                                primVarsX[2],
                                primVarsX[3],
                                centBx,
                                primVarsX[4],
                                primVarsX[5],
                                primVarsX[6]])
        #Update the full set of conservative variables
        consVarsAll = np.array([consVarsX[0],
                                consVarsX[1],
                                consVarsX[2],
                                consVarsX[3],
                                centBx,
                                consVarsX[4],
                                consVarsX[5],
                                consVarsX[6]])
        #Update the time and cycle number
        time += dt
        cycle += 1

        #Print the time, cycle number, and timestep size
        print("Cycle = " + str(cycle) + " Time = " + str(time) + " Dt = " + str(dt))

    #Return the final state of the primitive + conservative variables,
    #the face-centred magnetic fields, and the time and cycle when we stopped the simulation
    return primVarsAll, consVarsAll, faceBx, faceBy, faceBz, time, cycle, \
           secondLastPrimVars, secondLastConsVars, secondLastFaceBx, secondLastFaceBy, secondLastFaceBz, secondLastTime

