#TimestepCalculations_HelperModule.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides functions for calculating timestep sizes according to the Courant-Friedrichs-Lewy (CFL) formula.

#####IMPORT STATEMENTS######

#Import Numpy for matrix operations
import numpy as np

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants

####### HYDRODYNAMIC CARTESIAN TIMESTEP CALCULATORS #########

#Function: calcDt_1D_hydro
#Purpose: Calculates the timestep size for 1D hydrodynamic simulations.
#Input Parameters: consVars (the conservative variables for every cell in the
#                            simulation grid)
#                  gamma (the specific heat ratio for the ideal gas)
#                  cfl (the CFL number)
#                  dx (the cell width/size of the cell in the x-direction)
#Outputs: dt (the timestep size for the simulation cycle)
def calcDt_1D_hydro(consVars, gamma, cfl, dx):
    #Note: It would be a bit more straightforward to use the primitive variable matrix
    #      (which has velocity and pressure values) to calculate the timestep size.
    #      Here we use the conservative variables in order to obtain exactly the same
    #      timestep sizes as Athena, because having identical timesteps was critical
    #      for debugging purposes during PythonMHD's development.

    #Calculate the value of the specific heat ratio minus one
    gammaMinOne = gamma-1.0
    #Calculate the inverse density values for the entire simulation grid
    invDens = 1.0/consVars[0]
    #Calculate velocities from the momenta in the conservative variable matrix
    vx = consVars[1]*invDens #x-momenta -> x-velocity
    vy = consVars[2]*invDens #t-momenta -> y-velocity
    vz = consVars[3]*invDens #z-momenta -> z-velocity
    #Calculate the total velocity squared
    VSq = vx*vx + vy*vy + vz*vz
    #Calculate pressure as P = (gamma-1)*(E - 0.5*dens*Vsq)
    pres = np.maximum(gammaMinOne*(consVars[4]-0.5*consVars[0]*VSq),constants.SMALL_VALUE)
    #Calculate the hydrodynamic sound speed
    soundSpeedSq = gamma*pres*invDens
    #Calculate the max wavespeed in the x-direction
    maxWaveSpeedX = np.max(np.abs(vx) + np.sqrt(soundSpeedSq))
    #Calculate the inverse of how long it would take the
    #fastest wave to traverse the width of one cell
    maxInvDt = maxWaveSpeedX/dx
    #Our timestep size is equal to our CFL ratio (0 < cfl < 1)
    #times how long it takes the fastest wave to traverse one cell
    #(because we want the fastest wave to travel less than one grid cell
    # during one simulation cycle).
    dt_x = cfl/maxInvDt
    #Return the timestep size
    return dt_x

#Function: calcDt_2D_hydro
#Purpose: Calculates the timestep size for 2D hydrodynamic simulations.
#Input Parameters: consVars (the conservative variables for every cell in the
#                            simulation grid)
#                  gamma (the specific heat ratio for the ideal gas)
#                  cfl (the CFL number)
#                  dx (the cell width/size of the cell in the x-direction)
#                  dy (the cell height/size of the cell in the y-direction)
#Outputs: dt (the timestep size for the simulation cycle)
def calcDt_2D_hydro(consVars, gamma, cfl, dx, dy):
    # Note: It would be a bit more straightforward to use the primitive variable matrix
    #      (which has velocity and pressure values) to calculate the timestep size.
    #      Here we use the conservative variables in order to obtain exactly the same
    #      timestep sizes as Athena, because having identical timesteps was critical
    #      for debugging purposes during PythonMHD's development.

    #Calculate the value of the specific heat ratio minus one
    gammaMinOne = gamma - 1.0
    #Calculate the inverse density values for the entire simulation grid
    invDens = 1.0/consVars[0]
    #Calculate velocities from the momenta in the conservative variable matrix
    vx = consVars[1]*invDens #x-momenta -> x-velocity
    vy = consVars[2]*invDens #t-momenta -> y-velocity
    vz = consVars[3]*invDens #z-momenta -> z-velocity
    #Calculate the total velocity squared
    VSq = vx * vx + vy * vy + vz * vz
    #Calculate pressure as P = (gamma-1)*(E - 0.5*dens*Vsq)
    pres = np.maximum(gammaMinOne * (consVars[4] - 0.5 * consVars[0] * VSq), constants.SMALL_VALUE)
    #Calculate the hydrodynamic sound speed
    soundSpeedSq = gamma * pres * invDens
    #Calculate the max wavespeed in the x-direction
    maxWaveSpeedX = np.max(np.abs(vx) + np.sqrt(soundSpeedSq))
    #Calculate the max wavespeed in the y-direction
    maxWaveSpeedY = np.max(np.abs(vy) + np.sqrt(soundSpeedSq))
    #Calculate the inverse of how long it would take the
    #fastest wave in the x-direction to traverse the width of one cell
    maxInvDt_x = maxWaveSpeedX/dx
    #Calculate the inverse of how long it would take the
    #fastest wave in the y-direction to traverse the height of one cell
    maxInvDt_y = maxWaveSpeedY/dy

    #Our timestep size is equal to our CFL ratio (0 < cfl < 1)
    #times how long it takes the fastest wave to traverse one cell
    #(because we want the fastest wave to travel less than one grid cell
    #during one simulation cycle).

    #Calculate the timestep size for the x-direction.
    dt_x = cfl/maxInvDt_x
    # Calculate the timestep size for the y-direction.
    dt_y = cfl/maxInvDt_y
    #Return the smaller timestep size
    return np.minimum(dt_x, dt_y)

#Function: calcDt_3D_hydro
#Purpose: Calculates the timestep size for 3D hydrodynamic simulations.
#Input Parameters: consVars (the primitive variables for every cell in the
#                            simulation grid)
#                  gamma (the specific heat ratio for the ideal gas)
#                  cfl (the CFL number)
#                  dx (the cell width/size of the cell in the x-direction)
#                  dy (the cell height/size of the cell in the y-direction)
#                  dz (the cell depth/size of the cell in the z-direction)
#Outputs: dt (the timestep size for the simulation cycle)
def calcDt_3D_hydro(consVars, gamma, cfl, dx, dy, dz):
    # Note: It would be a bit more straightforward to use the primitive variable matrix
    #      (which has velocity and pressure values) to calculate the timestep size.
    #      Here we use the conservative variables in order to obtain exactly the same
    #      timestep sizes as Athena, because having identical timesteps was critical
    #      for debugging purposes during PythonMHD's development.

    #Calculate the value of the specific heat ratio minus one
    gammaMinOne = gamma - 1.0
    #Calculate the inverse density values for the entire simulation grid
    invDens = 1.0/consVars[0]
    #Calculate velocities from the momenta in the conservative variable matrix
    vx = consVars[1]*invDens #x-momenta -> x-velocity
    vy = consVars[2]*invDens #t-momenta -> y-velocity
    vz = consVars[3]*invDens #z-momenta -> z-velocity
    #Calculate the total velocity squared
    VSq = vx*vx + vy*vy + vz*vz
    #Calculate pressure as P = (gamma-1)*(E - 0.5*dens*Vsq)
    pres = np.maximum(gammaMinOne*(consVars[4] - 0.5*consVars[0]*VSq),constants.SMALL_VALUE)
    #Calculate the hydrodynamic sound speed
    soundSpeedSq = gamma*pres*invDens
    #Calculate the max wavespeed in the x-direction
    maxWaveSpeedX = np.max(np.abs(vx) + np.sqrt(soundSpeedSq))
    #Calculate the max wavespeed in the y-direction
    maxWaveSpeedY = np.max(np.abs(vy) + np.sqrt(soundSpeedSq))
    #Calculate the max wavespeed in the z-direction
    maxWaveSpeedZ = np.max(np.abs(vz) + np.sqrt(soundSpeedSq))
    #Calculate the inverse of how long it would take the
    #fastest wave in the x-direction to traverse the width of one cell
    maxInvDt_x = maxWaveSpeedX/dx
    #Calculate the inverse of how long it would take the
    #fastest wave in the y-direction to traverse the height of one cell
    maxInvDt_y = maxWaveSpeedY/dy
    #Calculate the inverse of how long it would take the
    #fastest wave in the z-direction to traverse the height of one cell
    maxInvDt_z = maxWaveSpeedZ/dz

    #Our timestep size is equal to our CFL ratio (0 < cfl < 1)
    #times how long it takes the fastest wave to traverse one cell
    #(because we want the fastest wave to travel less than one grid cell
    #during one simulation cycle).

    #Calculate the timestep size for the x-direction.
    dt_x = cfl/maxInvDt_x
    #Calculate the timestep size for the y-direction.
    dt_y = cfl/maxInvDt_y
    #Calculate the timestep size for the y-direction.
    dt_z = cfl/maxInvDt_z
    #Return the smaller timestep size
    return np.minimum(dt_x, np.minimum(dt_y,dt_z))

####### MHD CARTESIAN TIMESTEP CALCULATORS #########

#Function: calcDt_1D_mhd
#Purpose: Calculates the timestep size for 1D MHD simulations.
#Input Parameters: consVars (the conservative variables for every cell in the
#                            simulation grid, including all cell-centred magnetic
#                            field values)
#                  gamma (the specific heat ratio for the ideal gas)
#                  cfl (the CFL number)
#                  dx (the cell width/size of the cell in the x-direction)
#Outputs: dt (the timestep size for the simulation cycle)
def calcDt_1D_mhd(consVars, gamma, cfl, dx):
    #Note: It would be a bit more straightforward to use the primitive variable matrix
    #      (which has velocity and pressure values) to calculate the timestep size.
    #      Here we use the conservative variables in order to obtain exactly the same
    #      timestep sizes as Athena, because having identical timesteps was critical
    #      for debugging purposes during PythonMHD's development.

    #Calculate the value of the specific heat ratio minus one
    gammaMinOne = gamma-1.0
    #Calculate the inverse density values for the entire simulation grid
    invDens = 1.0/consVars[0]
    #Calculate velocities from the momenta in the conservative variable matrix
    vx = consVars[1]*invDens #x-momenta -> x-velocity
    vy = consVars[2]*invDens #t-momenta -> y-velocity
    vz = consVars[3]*invDens #z-momenta -> z-velocity
    #Calculate the total velocity squared
    VSq = vx*vx + vy*vy + vz*vz
    #Retrieve the magnetic field values that we will use for calculating the MHD wavespeeds
    Bx = consVars[4]
    By = consVars[5]
    Bz = consVars[6]
    #Calculate the total squared magnetic field intensity
    BSq = Bx*Bx + By*By + Bz*Bz
    #Calculate pressure as P = (gamma-1)*(E - 0.5*dens*Vsq)
    pres = np.maximum(gammaMinOne*(consVars[7]-0.5*consVars[0]*VSq - 0.5*BSq),constants.SMALL_VALUE)
    #Calculate the hydrodynamic sound speed
    soundSpeedSq = gamma*pres*invDens
    #Calculate the fast magnetosonic wavespeed in the x-direction
    speedSum = BSq*invDens + soundSpeedSq
    speedDiff = BSq*invDens - soundSpeedSq
    roeAvgFastMagSonSpeedSq = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(By*By+Bz*Bz)*invDens))
    #Calculate the max wavespeed in the x-direction
    maxWaveSpeedX = np.max(np.abs(vx) + np.sqrt(roeAvgFastMagSonSpeedSq))
    #Calculate the inverse of how long it would take the
    #fastest wave to traverse the width of one cell
    maxInvDt = maxWaveSpeedX/dx
    #Our timestep size is equal to our CFL ratio (0 < cfl < 1)
    #times how long it takes the fastest wave to traverse one cell
    #(because we want the fastest wave to travel less than one grid cell
    # during one simulation cycle).
    dt_x = cfl/maxInvDt
    #Return the timestep size
    return dt_x

#Function: calcDt_2D_mhd
#Purpose: Calculates the timestep size for 2D MHD simulations.
#Input Parameters: consVars (the conservative variables for every cell in the
#                            simulation grid, including all cell-centred magnetic
#                            field values)
#                  faceBx (the face-centred Bx values)
#                  faceBy (the face-centred By values)
#                  gamma (the specific heat ratio for the ideal gas)
#                  cfl (the CFL number)
#                  dx (the cell width/size of the cell in the x-direction)
#                  dy (the cell height/size of the cell in the y-direction)
#Outputs: dt (the timestep size for the simulation cycle)
def calcDt_2D_mhd(consVars, faceBx, faceBy, gamma, cfl, dx, dy):
    #Note: It would be a bit more straightforward to use the primitive variable matrix
    #      (which has velocity and pressure values) to calculate the timestep size.
    #      Here we use the conservative variables in order to obtain exactly the same
    #      timestep sizes as Athena, because having identical timesteps was critical
    #      for debugging purposes during PythonMHD's development.

    #Calculate the value of the specific heat ratio minus one
    gammaMinOne = gamma-1.0
    #Calculate the inverse density values for the entire simulation grid
    invDens = 1.0/consVars[0]
    #Calculate velocities from the momenta in the conservative variable matrix
    vx = consVars[1]*invDens #x-momenta -> x-velocity
    vy = consVars[2]*invDens #t-momenta -> y-velocity
    vz = consVars[3]*invDens #z-momenta -> z-velocity
    #Calculate the total velocity squared
    VSq = vx*vx + vy*vy + vz*vz
    #Retrieve the magnetic field values that we will use for calculating the MHD wavespeeds
    Bx = consVars[4] + np.abs(faceBx[:,0:faceBx.shape[1]-1] - consVars[4])
    By = consVars[5] + np.abs(faceBy[0:faceBy.shape[0]-1,:] - consVars[5])
    Bz = consVars[6]
    #Calculate the total squared magnetic field intensity
    BSq = Bx*Bx + By*By + Bz*Bz
    #Calculate pressure as P = (gamma-1)*(E - 0.5*dens*Vsq)
    pres = np.maximum(gammaMinOne*(consVars[7]-0.5*consVars[0]*VSq - 0.5*BSq),constants.SMALL_VALUE)
    #Calculate the hydrodynamic sound speed
    soundSpeedSq = gamma*pres*invDens
    #Calculate the fast magnetosonic wavespeed in the x- and y-directions
    speedSum = BSq*invDens + soundSpeedSq
    speedDiff = BSq*invDens - soundSpeedSq
    roeAvgFastMagSonSpeedSq_x = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(By*By+Bz*Bz)*invDens))
    roeAvgFastMagSonSpeedSq_y = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx+Bz*Bz)*invDens))
    #Calculate the max wavespeed in the x-direction
    maxWaveSpeedX = np.max(np.abs(vx) + np.sqrt(roeAvgFastMagSonSpeedSq_x))
    #Calculate the max wavespeed in the y-direction
    maxWaveSpeedY = np.max(np.abs(vy) + np.sqrt(roeAvgFastMagSonSpeedSq_y))
    #Calculate the inverse of how long it would take the
    #fastest wave in the x-direction to traverse the width of one cell
    maxInvDt_x = maxWaveSpeedX/dx
    #Calculate the inverse of how long it would take the
    #fastest wave in the y-direction to traverse the height of one cell
    maxInvDt_y = maxWaveSpeedY/dy
    #Use the larger inverse Dt value to calculate our timestep size
    maxInvDt = np.maximum(maxInvDt_x, maxInvDt_y)
    #Our timestep size is equal to our CFL ratio (0 < cfl < 1)
    #times how long it takes the fastest wave to traverse one cell
    #(because we want the fastest wave to travel less than one grid cell
    # during one simulation cycle).
    dt = cfl/maxInvDt
    #Return the timestep size
    return dt

#Function: calcDt_3D_mhd
#Purpose: Calculates the timestep size for 3D MHD simulations.
#Input Parameters: consVars (the conservative variables for every cell in the
#                            simulation grid, including all cell-centred magnetic
#                            field values)
#                  faceBx (the face-centred Bx values)
#                  faceBy (the face-centred By values)
#                  faceBz (the face-centred Bz values)
#                  gamma (the specific heat ratio for the ideal gas)
#                  cfl (the CFL number)
#                  dx (the cell width/size of the cell in the x-direction)
#                  dy (the cell height/size of the cell in the y-direction)
#                  dz (the cell depth/size of the cell in the z-direction)
#Outputs: dt (the timestep size for the simulation cycle)
def calcDt_3D_mhd(consVars, faceBx, faceBy, faceBz, gamma, cfl, dx, dy, dz):
    #Note: It would be a bit more straightforward to use the primitive variable matrix
    #      (which has velocity and pressure values) to calculate the timestep size.
    #      Here we use the conservative variables in order to obtain exactly the same
    #      timestep sizes as Athena, because having identical timesteps was critical
    #      for debugging purposes during PythonMHD's development.

    #Calculate the value of the specific heat ratio minus one
    gammaMinOne = gamma-1.0
    #Calculate the inverse density values for the entire simulation grid
    invDens = 1.0/consVars[0]
    #Calculate velocities from the momenta in the conservative variable matrix
    vx = consVars[1]*invDens #x-momenta -> x-velocity
    vy = consVars[2]*invDens #t-momenta -> y-velocity
    vz = consVars[3]*invDens #z-momenta -> z-velocity
    #Calculate the total velocity squared
    VSq = vx*vx + vy*vy + vz*vz
    #Retrieve the magnetic field values that we will use for calculating the MHD wavespeeds
    Bx = consVars[4] + np.abs(faceBx[:,0:faceBx.shape[1]-1,:] - consVars[4])
    By = consVars[5] + np.abs(faceBy[0:faceBy.shape[0]-1,:,:] - consVars[5])
    Bz = consVars[6] + np.abs(faceBz[:,:,0:faceBz.shape[2]-1] - consVars[6])
    #Calculate the total squared magnetic field intensity
    BSq = Bx*Bx + By*By + Bz*Bz
    #Calculate pressure as P = (gamma-1)*(E - 0.5*dens*Vsq)
    pres = np.maximum(gammaMinOne*(consVars[7]-0.5*consVars[0]*VSq - 0.5*BSq),constants.SMALL_VALUE)
    #Calculate the hydrodynamic sound speed
    soundSpeedSq = gamma*pres*invDens
    #Calculate the fast magnetosonic wavespeed in the x- and y-directions
    speedSum = BSq*invDens + soundSpeedSq
    speedDiff = BSq*invDens - soundSpeedSq
    roeAvgFastMagSonSpeedSq_x = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(By*By+Bz*Bz)*invDens))
    roeAvgFastMagSonSpeedSq_y = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx+Bz*Bz)*invDens))
    roeAvgFastMagSonSpeedSq_z = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx+By*By)*invDens))
    #Calculate the max wavespeed in the x-direction
    maxWaveSpeedX = np.max(np.abs(vx) + np.sqrt(roeAvgFastMagSonSpeedSq_x))
    #Calculate the max wavespeed in the y-direction
    maxWaveSpeedY = np.max(np.abs(vy) + np.sqrt(roeAvgFastMagSonSpeedSq_y))
    #Calculate the max wavespeed in the z-direction
    maxWaveSpeedZ = np.max(np.abs(vz) + np.sqrt(roeAvgFastMagSonSpeedSq_z))
    #Calculate the inverse of how long it would take the
    #fastest wave in the x-direction to traverse the width of one cell
    maxInvDt_x = maxWaveSpeedX/dx
    #Calculate the inverse of how long it would take the
    #fastest wave in the y-direction to traverse the height of one cell
    maxInvDt_y = maxWaveSpeedY/dy
    #Calculate the inverse of how long it would take the
    #fastest wave in the z-direction to traverse the depth of one cell
    maxInvDt_z = maxWaveSpeedZ/dz
    #Use the larger inverse Dt value to calculate our timestep size
    maxInvDt = np.maximum(maxInvDt_x, np.maximum(maxInvDt_y, maxInvDt_z))
    #Our timestep size is equal to our CFL ratio (0 < cfl < 1)
    #times how long it takes the fastest wave to traverse one cell
    #(because we want the fastest wave to travel less than one grid cell
    # during one simulation cycle).
    dt = cfl/maxInvDt
    #Return the timestep size
    return dt
#
# #Function: calcDt_2D_mhd
# #Purpose: Calculates the timestep size for 2D MHD simulations.
# #Input Parameters: primVars (the primitive variables for every cell in the
# #                            simulation grid)
# #                  gamma (the specific heat ratio for the ideal gas)
# #                  cfl (the CFL number)
# #                  dx (the cell width/size of the cell in the x-direction)
# #                  dy (the cell height/size of the cell in the y-direction)
# #                  dz (the cell depth/size of the cell in the z-direction)
# #Outputs: dt (the timestep size for the simulation cycle)
# def calcDt_2D_mhd(primVars, energy, faceBx, faceBy, gamma, cfl, dx, dy):
#     Bx = primVars[4] + np.abs(faceBx[:, 0:faceBx.shape[1] - 1] - primVars[4])
#     By = primVars[5] + np.abs(faceBy[0:faceBy.shape[0]-1,:]-primVars[5])
#     Bz = primVars[6]
#     #Divide the total magnetic field intensity by the square root of the density in each cell
#     reducedB = np.sqrt((Bx*Bx + By*By + Bz*Bz)/primVars[0,:])
#     pressure = (gamma-1.0)*(energy
#                             - 0.5*primVars[0]*(primVars[1]*primVars[1]+primVars[2]*primVars[2]+primVars[3]*primVars[3])
#                             - 0.5*(Bx*Bx+By*By+Bz*Bz))
#     # Calculate the hydrodynamic sound speed
#     soundSpeed = np.sqrt(gamma * pressure / primVars[0])
#     #Calculate the full sound speed (hydro + magnetic)
#     soundSpeedStar = np.sqrt(np.square(soundSpeed) + np.square(reducedB))
#     soundSpeedSq = gamma*pressure/primVars[0]
#     #Calculate the Alfven speeds in the x, y, and z directions
#     alfvenSpeedX = Bx/np.sqrt(primVars[0])
#     alfvenSpeedY = By/np.sqrt(primVars[0])
#     alfvenSpeedZ = Bz/np.sqrt(primVars[0])
#     speedSum = np.square(reducedB) + soundSpeedSq
#     speedDiff = np.square(reducedB) - soundSpeedSq
#     #Calculate the fast magnetosonic speeds in the x, y, and z directions
#     fastMagSonSpeedX = np.sqrt(0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(By*By + Bz*Bz)/primVars[0])))
#     fastMagSonSpeedY = np.sqrt(0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx + Bz*Bz)/primVars[0])))
#     fastMagSonSpeedZ = np.sqrt(0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx + By*By)/primVars[0])))
#     fastMagSonSpeedXSq = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(By*By + Bz*Bz)/primVars[0]))
#     fastMagSonSpeedYSq = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx + Bz*Bz)/primVars[0]))
#     fastMagSonSpeedZSq = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx + By*By)/primVars[0]))
#     #Find the max wavespeed in the x, y, and z directions
#     maxWaveSpeedX = np.max(np.abs(primVars[1,:]) + fastMagSonSpeedX)
#     maxWaveSpeedY = np.max(np.abs(primVars[2,:]) + fastMagSonSpeedY)
#     maxWaveSpeedXPos = (np.abs(primVars[1,:]) + fastMagSonSpeedX) == maxWaveSpeedX
#     maxWaveSpeedYPos = (np.abs(primVars[2, :]) + fastMagSonSpeedY) == maxWaveSpeedY
#     #Calculate a timestep based on the max wavespeed in the x-direction
#     dt_x = cfl*dx/maxWaveSpeedX
#     #Calculate a timestep based on the max wavespeed in the y-direction
#     dt_y = cfl*dy/maxWaveSpeedY
#     dt = np.minimum(dt_x, dt_y)
#     #Return the smallest timestep size
#     return dt
#
# #Function: calcDt_3D_mhd
# #Purpose: Calculates the timestep size for 3D MHD simulations.
# #Input Parameters: consVars (the conservative variables for every cell in the
# #                            simulation grid)
# #                  gamma (the specific heat ratio for the ideal gas)
# #                  cfl (the CFL number)
# #                  dx (the cell width/size of the cell in the x-direction)
# #                  dy (the cell height/size of the cell in the y-direction)
# #                  dz (the cell depth/size of the cell in the z-direction)
# #Outputs: dt (the timestep size for the simulation cycle)
# def calcDt_3D_mhd(primVars, energy,faceBx, faceBy, faceBz, gamma, cfl, dx, dy, dz):
#     Bx = primVars[4] + np.abs(faceBx[:,0:faceBx.shape[1]-1,:]-primVars[4])
#     By = primVars[5] + np.abs(faceBy[0:faceBy.shape[0]-1,:,:]-primVars[5])
#     Bz = primVars[6] + np.abs(faceBz[:,:,0:faceBz.shape[2]-1]-primVars[6])
#     #Divide the total magnetic field intensity by the square root of the density in each cell
#     reducedB = np.sqrt((Bx*Bx + By*By + Bz*Bz)/primVars[0,:])
#     pressure = (gamma-1.0)*(energy
#                             - 0.5*primVars[0]*(primVars[1]*primVars[1]+primVars[2]*primVars[2]+primVars[3]*primVars[3])
#                             - 0.5*(Bx*Bx+By*By+Bz*Bz))
#     # Calculate the hydrodynamic sound speed
#     soundSpeed = np.sqrt(gamma * pressure / primVars[0])
#     #Calculate the full sound speed (hydro + magnetic)
#     soundSpeedStar = np.sqrt(np.square(soundSpeed) + np.square(reducedB))
#     soundSpeedSq = gamma*pressure/primVars[0]
#     #Calculate the Alfven speeds in the x, y, and z directions
#     alfvenSpeedX = Bx/np.sqrt(primVars[0])
#     alfvenSpeedY = By/np.sqrt(primVars[0])
#     alfvenSpeedZ = Bz/np.sqrt(primVars[0])
#     speedSum = np.square(reducedB) + soundSpeedSq
#     speedDiff = np.square(reducedB) - soundSpeedSq
#     #Calculate the fast magnetosonic speeds in the x, y, and z directions
#     fastMagSonSpeedX = np.sqrt(0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(By*By + Bz*Bz)/primVars[0])))
#     fastMagSonSpeedY = np.sqrt(0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx + Bz*Bz)/primVars[0])))
#     fastMagSonSpeedZ = np.sqrt(0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx + By*By)/primVars[0])))
#     fastMagSonSpeedXSq = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(By*By + Bz*Bz)/primVars[0]))
#     fastMagSonSpeedYSq = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx + Bz*Bz)/primVars[0]))
#     fastMagSonSpeedZSq = 0.5*(speedSum + np.sqrt(speedDiff*speedDiff + 4.0*soundSpeedSq*(Bx*Bx + By*By)/primVars[0]))
#     #Find the max wavespeed in the x, y, and z directions
#     maxWaveSpeedX = np.max(np.abs(primVars[1,:]) + fastMagSonSpeedX)
#     testX = np.abs(primVars[1,:]) + fastMagSonSpeedX
#     maxWaveSpeedY = np.max(np.abs(primVars[2,:]) + fastMagSonSpeedY)
#     maxWaveSpeedZ = np.max(np.abs(primVars[3,:]) + fastMagSonSpeedZ)
#
#     #Calculate a timestep based on the max wavespeed in the x-direction
#     dt_x = cfl*dx/maxWaveSpeedX
#     #Calculate a timestep based on the max wavespeed in the y-direction
#     dt_y = cfl*dy/maxWaveSpeedY
#     #Calculate a timestep based on the max wavespeed in the z-direction
#     dt_z = cfl*dz/maxWaveSpeedZ
#     dt = np.minimum(dt_x, np.minimum(dt_y, dt_z))
#     #Return the smallest timestep size
#     return dt
