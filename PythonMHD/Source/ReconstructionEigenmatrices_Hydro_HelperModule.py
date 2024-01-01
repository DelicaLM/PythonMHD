#ReconstructionEigenmatrices_Hydro_HelperModule.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides functions for calculating the left and right eigenmatrices that are
#         required in the primitive variable reconstruction algorithms (PLM and PPM).
#         This module provides the primitive variable eigenmatrices for hydrodynamic
#         simulations, using the same formulas as the 2017 version of Athena [1,2].
#References:
# 1. https://github.com/PrincetonUniversity/Athena-Cversion
# 2. Stone, J. M., Gardiner, T. A., Teuben, P., Hawley, J. F., & Simon, J. B. (2008).
#    Athena: A new code for astrophysical MHD. The Astrophysical Journal Supplemental Series,
#    178(1), 137-177. https://iopscience.iop.org/article/10.1086/588755/pdf.

#Import NumPy for matrix operations
import numpy as np

#Function: getEigenmatricesX_hydro
#Purpose: Calculates the x-direction eigenvalues and eigenmatrices for every
#         cell in the simulation grid.
#Input Parameters: primVars (the primitive variables for every cell in the grid)
#                  gamma (the specific heat ratio for the ideal gas
#Outputs: eigenVals (the eigenvalues for every cell in the simulation grid)
#         leftEigenmatrix (the left eigenmatrix for every cell in the simulation grid)
#         rightEigenmatrix (the right eigenmatrix for every cell in the simulation grid)
def getEigenmatricesX_hydro(primVars, gamma):
    #Create matrices for the eigenvalues and eigenmatrices
    if len(primVars.shape) == 2: #if the sim is 1D
        eigenVals = np.zeros(shape=(5,primVars.shape[1]))
        leftEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1]))
        rightEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1]))
    elif len(primVars.shape) == 3: #if the sim is 2D
        eigenVals = np.zeros(shape=(5,primVars.shape[1],primVars.shape[2]))
        leftEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2]))
        rightEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2]))
    else: #if the sim is 3D
        eigenVals = np.zeros(shape=(5,primVars.shape[1], primVars.shape[2],primVars.shape[3]))
        leftEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2],primVars.shape[3]))
        rightEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2],primVars.shape[3]))

    #Calculate the hydrodynamic sound speed
    gammaPresProd = gamma*primVars[4]
    soundSpeedSq = gammaPresProd/primVars[0]
    soundSpeed = np.sqrt(soundSpeedSq)

    #Calculate the five eigenvalues/wavespeeds
    #that we need for hydrodynamic simulations
    eigenVals[0] = primVars[1] - soundSpeed #vx - cs
    eigenVals[1] = primVars[1] #vx
    eigenVals[2] = primVars[1] #vx
    eigenVals[3] = primVars[1] #vx
    eigenVals[4] = primVars[1] + soundSpeed #vx + cs

    #Add values to the right eigenmatrix
    #(eigenvectors are stored as columns)
    rightEigenmatrix[0,0,:] = 1.0
    rightEigenmatrix[1,0,:] = -soundSpeed/primVars[0]
    rightEigenmatrix[4,0,:] = soundSpeedSq

    rightEigenmatrix[0,1,:] = 1.0

    rightEigenmatrix[2,2,:] = 1.0

    rightEigenmatrix[3,3,:] = 1.0

    rightEigenmatrix[0,4,:] = 1.0
    rightEigenmatrix[1,4,:] = -rightEigenmatrix[1,0]
    rightEigenmatrix[4,4,:] = soundSpeedSq

    #Add values to the left eigenmatrix
    #(eigenvectors are stored as rows)
    leftEigenmatrix[0,1,:] = -0.5*primVars[0]/soundSpeed
    leftEigenmatrix[0,4,:] = 0.5/soundSpeedSq

    leftEigenmatrix[1,0,:] = 1.0
    leftEigenmatrix[1,4,:] = -1.0/soundSpeedSq

    leftEigenmatrix[2,2,:] = 1.0

    leftEigenmatrix[3,3,:] = 1.0

    leftEigenmatrix[4,1,:] = -leftEigenmatrix[0,1]
    leftEigenmatrix[4,4,:] = leftEigenmatrix[0,4]

    #Return the eigenvalues and eigenmatrices
    return eigenVals, leftEigenmatrix, rightEigenmatrix

#Function: getEigenmatricesY_hydro
#Purpose: Calculates the y-direction eigenvalues and eigenmatrices for every
#         cell in the simulation grid.
#Input Parameters: primVars (the primitive variables for every cell in the grid)
#                  gamma (the specific heat ratio for the ideal gas
#Outputs: eigenVals (the eigenvalues for every cell in the simulation grid)
#         leftEigenmatrix (the left eigenmatrix for every cell in the simulation grid)
#         rightEigenmatrix (the right eigenmatrix for every cell in the simulation grid)
def getEigenmatricesY_hydro(primVars, gamma):
    #Create matrices for the eigenvalues and eigenmatrices
    if len(primVars.shape) == 3: #if the sim is 2D
        eigenVals = np.zeros(shape=(5,primVars.shape[1],primVars.shape[2]))
        leftEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2]))
        rightEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2]))
    else: #if the sim is 3D
        eigenVals = np.zeros(shape=(5,primVars.shape[1], primVars.shape[2],primVars.shape[3]))
        leftEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2],primVars.shape[3]))
        rightEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2],primVars.shape[3]))

    #Calculate the hydrodynamic sound speed
    gammaPresProd = gamma*primVars[4]
    soundSpeedSq = gammaPresProd/primVars[0]
    soundSpeed = np.sqrt(soundSpeedSq)

    #Calculate the five eigenvalues/wavespeeds
    #that we need for hydrodynamic simulations
    eigenVals[0] = primVars[2] - soundSpeed #vy - cs
    eigenVals[1] = primVars[2] #vy
    eigenVals[2] = primVars[2] #vy
    eigenVals[3] = primVars[2] #vy
    eigenVals[4] = primVars[2] + soundSpeed #vy + cs

    #Add values to the right eigenmatrix
    #(eigenvectors are stored as columns)
    rightEigenmatrix[0,0,:] = 1.0
    rightEigenmatrix[2,0,:] = -soundSpeed/primVars[0]
    rightEigenmatrix[4,0,:] = soundSpeedSq

    rightEigenmatrix[0,2,:] = 1.0

    rightEigenmatrix[1,1,:] = 1.0

    rightEigenmatrix[3,3,:] = 1.0

    rightEigenmatrix[0,4,:] = 1.0
    rightEigenmatrix[2,4,:] = -rightEigenmatrix[2,0]
    rightEigenmatrix[4,4,:] = soundSpeedSq

    #Add values to the left eigenmatrix
    #(eigenvectors are stored as rows)
    leftEigenmatrix[0,2,:] = -0.5*primVars[0]/soundSpeed
    leftEigenmatrix[0,4,:] = 0.5/soundSpeedSq

    leftEigenmatrix[2,0,:] = 1.0
    leftEigenmatrix[2,4,:] = -1.0/soundSpeedSq

    leftEigenmatrix[1,1,:] = 1.0

    leftEigenmatrix[3,3,:] = 1.0

    leftEigenmatrix[4,2,:] = -leftEigenmatrix[0,2]
    leftEigenmatrix[4,4,:] = leftEigenmatrix[0,4]

    #Return the eigenvalues and eigenmatrices
    return eigenVals, leftEigenmatrix, rightEigenmatrix

# #Function: getEigenmatricesY_hydro
# #Purpose: Calculates the y-direction eigenvalues and eigenmatrices for every
# #         cell in the simulation grid.
# #Input Parameters: primVars (the primitive variables for every cell in the grid)
# #                  gamma (the specific heat ratio for the ideal gas
# #Outputs: eigenVals (the eigenvalues for every cell in the simulation grid)
# #         leftEigenmatrix (the left eigenmatrix for every cell in the simulation grid)
# #         rightEigenmatrix (the right eigenmatrix for every cell in the simulation grid)
# def getEigenmatricesY_hydro(primVars, gamma):
#     #Create matrices for the eigenvalues and eigenmatrices
#     if len(primVars.shape) == 2: #if the sim is 1D
#         eigenVals = np.zeros(shape=(5,primVars.shape[1]))
#         leftEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1]))
#         rightEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1]))
#     elif len(primVars.shape) == 3: #if the sim is 2D
#         eigenVals = np.zeros(shape=(5,primVars.shape[1],primVars.shape[2]))
#         leftEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2]))
#         rightEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2]))
#     else: #if the sim is 3D
#         eigenVals = np.zeros(shape=(5,primVars.shape[1], primVars.shape[2],primVars.shape[3]))
#         leftEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2],primVars.shape[3]))
#         rightEigenmatrix = np.zeros(shape=(5,5,primVars.shape[1],primVars.shape[2],primVars.shape[3]))
#
#     #Calculate the hydrodynamic sound speed
#     soundSpeed = np.sqrt(gamma*primVars[4]/primVars[0])
#
#     #Calculate the five wavespeeds/eigenvalues
#     #that we need for hydrodynamic simulations
#     eigenVals[0] = primVars[2] - soundSpeed #vy - cs
#     eigenVals[1] = primVars[2] #vy
#     eigenVals[2] = primVars[2] #vy
#     eigenVals[3] = primVars[2] #vy
#     eigenVals[4] = primVars[2] + soundSpeed #vy + cs
#
#     #Add values to the right eigenmatrix
#     #(eigenvectors are stored as columns)
#     rightEigenmatrix[0,0,:] = 1.0
#     rightEigenmatrix[1,0,:] = -soundSpeed/primVars[0]
#     rightEigenmatrix[4,0,:] = soundSpeed*soundSpeed
#
#     rightEigenmatrix[0,1,:] = 1.0
#
#     rightEigenmatrix[2,2,:] = 1.0
#
#     rightEigenmatrix[3,3,:] = 1.0
#
#     rightEigenmatrix[0,4,:] = 1.0
#     rightEigenmatrix[1,4,:] = -rightEigenmatrix[1,0]
#     rightEigenmatrix[4,4,:] = soundSpeed*soundSpeed
#
#     #Add values to the left eigenmatrix
#     #(eigenvectors are stored as rows)
#     leftEigenmatrix[0,1,:] = -0.5*primVars[0]/soundSpeed
#     leftEigenmatrix[0,4,:] = 0.5/(soundSpeed*soundSpeed)
#
#     leftEigenmatrix[1,0,:] = 1.0
#     leftEigenmatrix[1,4,:] = -1.0/(soundSpeed*soundSpeed)
#
#     leftEigenmatrix[2,2,:] = 1.0
#
#     leftEigenmatrix[3,3,:] = 1.0
#
#     leftEigenmatrix[4,1,:] = -leftEigenmatrix[0,1]
#     leftEigenmatrix[4,4,:] = leftEigenmatrix[0,4]
#
#     #Return the eigenvalues and eigenmatrices
#     return eigenVals, leftEigenmatrix, rightEigenmatrix


#Function: getEigenmatricesZ_hydro
#Purpose: Calculates the z-direction eigenvalues and eigenmatrices for every
#         cell in the simulation grid.
#Input Parameters: primVars (the primitive variables for every cell in the grid)
#                  gamma (the specific heat ratio for the ideal gas
#Outputs: eigenVals (the eigenvalues for every cell in the simulation grid)
#         leftEigenmatrix (the left eigenmatrix for every cell in the simulation grid)
#         rightEigenmatrix (the right eigenmatrix for every cell in the simulation grid)
def getEigenmatricesZ_hydro(primVars, gamma):
    #Create matrices for the eigenvalues and eigenmatrices
    eigenVals = np.zeros(shape=(5, primVars.shape[1], primVars.shape[2], primVars.shape[3]))
    leftEigenmatrix = np.zeros(shape=(5, 5, primVars.shape[1], primVars.shape[2], primVars.shape[3]))
    rightEigenmatrix = np.zeros(shape=(5, 5, primVars.shape[1], primVars.shape[2], primVars.shape[3]))

    #Calculate the hydrodynamic sound speed
    gammaPresProd = gamma*primVars[4]
    soundSpeedSq = gammaPresProd/primVars[0]
    soundSpeed = np.sqrt(soundSpeedSq)

    #Calculate the five eigenvalues/wavespeeds
    #that we need for hydrodynamic simulations
    eigenVals[0] = primVars[3] - soundSpeed #vz - cs
    eigenVals[1] = primVars[3] #vz
    eigenVals[2] = primVars[3] #vz
    eigenVals[3] = primVars[3] #vz
    eigenVals[4] = primVars[3] + soundSpeed #vz + cs

    #Add values to the right eigenmatrix
    #(eigenvectors are stored as columns)
    rightEigenmatrix[0,0,:] = 1.0
    rightEigenmatrix[3,0,:] = -soundSpeed/primVars[0]
    rightEigenmatrix[4,0,:] = soundSpeedSq

    rightEigenmatrix[0,3,:] = 1.0

    rightEigenmatrix[1,1,:] = 1.0

    rightEigenmatrix[2,2,:] = 1.0

    rightEigenmatrix[0,4,:] = 1.0
    rightEigenmatrix[3,4,:] = -rightEigenmatrix[3,0]
    rightEigenmatrix[4,4,:] = soundSpeedSq

    #Add values to the left eigenmatrix
    #(eigenvectors are stored as rows)
    leftEigenmatrix[0,3,:] = -0.5*primVars[0]/soundSpeed
    leftEigenmatrix[0,4,:] = 0.5/soundSpeedSq

    leftEigenmatrix[3,0,:] = 1.0
    leftEigenmatrix[3,4,:] = -1.0/soundSpeedSq

    leftEigenmatrix[1,1,:] = 1.0

    leftEigenmatrix[2,2,:] = 1.0

    leftEigenmatrix[4,3,:] = -leftEigenmatrix[0,3]
    leftEigenmatrix[4,4,:] = leftEigenmatrix[0,4]

    #Return the eigenvalues and eigenmatrices
    return eigenVals, leftEigenmatrix, rightEigenmatrix