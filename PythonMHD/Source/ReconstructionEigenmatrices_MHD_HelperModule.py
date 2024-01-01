#ReconstructionEigenmatrices_MHD_HelperModule.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides functions for calculating the left and right eigenmatrices that are
#         required in the primitive variable reconstruction algorithms (PLM and PPM).
#         This module provides the primitive variable eigenmatrices for magnetohydrodynamic
#         simulations, using the same formulas as the 2017 version of Athena [1,2].
#References:
# 1. https://github.com/PrincetonUniversity/Athena-Cversion
# 2. Stone, J. M., Gardiner, T. A., Teuben, P., Hawley, J. F., & Simon, J. B. (2008).
#    Athena: A new code for astrophysical MHD. The Astrophysical Journal Supplemental Series,
#    178(1), 137-177. https://iopscience.iop.org/article/10.1086/588755/pdf.

#Import NumPy for matrix operations
import numpy as np

#Function: getEigenmatricesX_mhd
#Purpose: Calculates the x-direction eigenvalues and eigenmatrices for every
#         cell in the simulation grid.
#Input Parameters: primVarsX (the primitive variables for every cell in the grid)
#                  Bx (the cell-centred x-component of the magnetic field for
#                      every cell in the simulation grid)
#                  gamma (the specific heat ratio for the ideal gas
#Outputs: eigenVals (the eigenvalues for every cell in the simulation grid)
#         leftEigenmatrix (the left eigenmatrix for every cell in the simulation grid)
#         rightEigenmatrix (the right eigenmatrix for every cell in the simulation grid)
def getEigenmatricesX_mhd(primVarsX, Bx, gamma):
    #Create matrices for the eigenvalues and eigenmatrices
    if len(primVarsX.shape) == 2: #if the sim is 1D
        eigenVals = np.zeros(shape=(7,primVarsX.shape[1]))
        leftEigenmatrix = np.zeros(shape=(7,7,primVarsX.shape[1]))
        rightEigenmatrix = np.zeros(shape=(7,7,primVarsX.shape[1]))
    elif len(primVarsX.shape) == 3: #if the sim is 2D
        eigenVals = np.zeros(shape=(7,primVarsX.shape[1],primVarsX.shape[2]))
        leftEigenmatrix = np.zeros(shape=(7,7,primVarsX.shape[1],primVarsX.shape[2]))
        rightEigenmatrix = np.zeros(shape=(7,7,primVarsX.shape[1],primVarsX.shape[2]))
    else: #if the sim is 3D
        eigenVals = np.zeros(shape=(7,primVarsX.shape[1],primVarsX.shape[2],primVarsX.shape[3]))
        leftEigenmatrix = np.zeros(shape=(7,7,primVarsX.shape[1],primVarsX.shape[2],primVarsX.shape[3]))
        rightEigenmatrix = np.zeros(shape=(7,7,primVarsX.shape[1],primVarsX.shape[2],primVarsX.shape[3]))

    #Calculate the transverse magnetic field intensity
    BtranSq = np.square(primVarsX[4]) + np.square(primVarsX[5])
    Btran = np.sqrt(BtranSq)
    gammaPresProduct = gamma*primVarsX[6]
    inverseDensity = 1.0/primVarsX[0]
    sqrtDens = np.sqrt(primVarsX[0])
    reducedBtranSq = BtranSq*inverseDensity
    #Calculate the hydrodynamic sound speed
    soundSpeedSq = gammaPresProduct*inverseDensity
    soundSpeed = np.sqrt(soundSpeedSq)
    #Calculate the Alfven speed in the x-direction
    alfvenSpeedSq = Bx*Bx*inverseDensity
    alfvenSpeed = np.sqrt(alfvenSpeedSq)

    #Calculate the fast magnetosonic speed
    speedSum = alfvenSpeedSq + reducedBtranSq + soundSpeedSq
    speedDiff = alfvenSpeedSq + reducedBtranSq - soundSpeedSq
    speedSqrt = np.sqrt((speedDiff*speedDiff
                         + 4.0*soundSpeedSq*reducedBtranSq).astype(np.double))
    fastMagSonSpeedSq = 0.5*(speedSum + speedSqrt)
    fastMagSonSpeed = np.sqrt(fastMagSonSpeedSq)

    #Calculate the slow magnetosonic speed
    slowMagSonSpeedSq = soundSpeedSq*alfvenSpeedSq/fastMagSonSpeedSq
    slowMagSonSpeed = np.sqrt(slowMagSonSpeedSq)

    #Calculate the BetaY and BetaZ coefficients
    betaY = np.zeros(shape=Btran.shape)
    betaZ = np.zeros(shape=Btran.shape)
    betaY[Btran == 0.0] = 1.0
    betaY[Btran > 0.0] = primVarsX[4][Btran > 0.0]/Btran[Btran > 0.0]
    betaZ[Btran > 0.0] = primVarsX[5][Btran > 0.0]/Btran[Btran > 0.0]

    #Calculate the fastAlpha and slowAlpha coefficients
    fastAlpha = np.zeros(shape=betaY.shape)
    slowAlpha = np.zeros(shape=betaY.shape)
    fastAlphaOneIndices = speedSqrt == 0.0
    fastAlpha[fastAlphaOneIndices] = 1.0
    slowAlphaOneIndices = np.logical_and(np.logical_not(fastAlphaOneIndices), (soundSpeedSq - slowMagSonSpeedSq) <= 0.0)
    slowAlpha[slowAlphaOneIndices] = 1.0
    secondFastAlphaOneIndices = np.logical_and(np.logical_not(np.logical_or(fastAlphaOneIndices,slowAlphaOneIndices)),
                                               (fastMagSonSpeedSq-soundSpeedSq) <= 0.0)
    fastAlpha[secondFastAlphaOneIndices] = 1.0
    alphaCalcIndices = np.logical_not(np.logical_or(fastAlphaOneIndices,
                                                    np.logical_or(slowAlphaOneIndices,
                                                                  secondFastAlphaOneIndices)))
    fastAlpha[alphaCalcIndices] = np.sqrt((soundSpeedSq
                                           - slowMagSonSpeedSq)[alphaCalcIndices]
                                          / speedSqrt[alphaCalcIndices])

    slowAlpha[alphaCalcIndices] = np.sqrt((fastMagSonSpeedSq
                                           - soundSpeedSq)[alphaCalcIndices]
                                          / speedSqrt[alphaCalcIndices])
    #Get the sign of Bx in every cell
    BxSign = np.sign(Bx)
    BxSign[BxSign == 0.0] = 1.0
    #Calculate additional coefficients that will simplify the eigenvector calculations
    coeffQf = fastMagSonSpeed*fastAlpha*BxSign
    coeffQs = slowMagSonSpeed*slowAlpha*BxSign
    coeffAf = soundSpeed*fastAlpha*sqrtDens
    coeffAs = soundSpeed*slowAlpha*sqrtDens

    #Calculate the seven wavespeed eigenvalues
    eigenVals[0,:] = primVarsX[1] - fastMagSonSpeed #vx - cf
    eigenVals[1,:] = primVarsX[1] - alfvenSpeed #vx - ca
    eigenVals[2,:] = primVarsX[1] - slowMagSonSpeed #vx - cs
    eigenVals[3,:] = primVarsX[1] #vx
    eigenVals[4,:] = primVarsX[1] + slowMagSonSpeed #vx + cs
    eigenVals[5,:] = primVarsX[1] + alfvenSpeed #vx + ca
    eigenVals[6,:] = primVarsX[1] + fastMagSonSpeed #vx + cf

    #Add values to the right eigenmatrix
    rightEigenmatrix[0,0,:] = primVarsX[0]*fastAlpha
    rightEigenmatrix[0,2,:] = primVarsX[0]*slowAlpha
    rightEigenmatrix[0,3,:] = 1.0
    rightEigenmatrix[0,4,:] = rightEigenmatrix[0,2]
    rightEigenmatrix[0,6,:] = rightEigenmatrix[0,0]

    rightEigenmatrix[1,0,:] = -fastMagSonSpeed*fastAlpha
    rightEigenmatrix[1,2,:] = -slowMagSonSpeed*slowAlpha
    rightEigenmatrix[1,4,:] = -rightEigenmatrix[1,2]
    rightEigenmatrix[1,6,:] = -rightEigenmatrix[1,0]

    rightEigenmatrix[2,0,:] = coeffQs*betaY
    rightEigenmatrix[2,1,:] = -betaZ
    rightEigenmatrix[2,2,:] = -coeffQf*betaY
    rightEigenmatrix[2,4,:] = -rightEigenmatrix[2,2]
    rightEigenmatrix[2,5,:] = betaZ
    rightEigenmatrix[2,6,:] = -rightEigenmatrix[2,0]

    rightEigenmatrix[3,0,:] = coeffQs*betaZ
    rightEigenmatrix[3,1,:] = betaY
    rightEigenmatrix[3,2,:] = -coeffQf*betaZ
    rightEigenmatrix[3,4,:] = -rightEigenmatrix[3,2]
    rightEigenmatrix[3,5,:] = -betaY
    rightEigenmatrix[3,6,:] = -rightEigenmatrix[3,0]

    rightEigenmatrix[6,0,:] = primVarsX[0]*soundSpeedSq*fastAlpha
    rightEigenmatrix[6,2,:] = primVarsX[0]*soundSpeedSq*slowAlpha
    rightEigenmatrix[6,4,:] = rightEigenmatrix[6,2]
    rightEigenmatrix[6,6,:] = rightEigenmatrix[6,0]

    rightEigenmatrix[4,0,:] = coeffAs*betaY
    rightEigenmatrix[4,1,:] = -betaZ*BxSign*sqrtDens
    rightEigenmatrix[4,2,:] = -coeffAf*betaY
    rightEigenmatrix[4,4,:] = rightEigenmatrix[4,2]
    rightEigenmatrix[4,5,:] = rightEigenmatrix[4,1]
    rightEigenmatrix[4,6,:] = rightEigenmatrix[4,0]

    rightEigenmatrix[5,0,:] = coeffAs*betaZ
    rightEigenmatrix[5,1,:] = betaY*BxSign*sqrtDens
    rightEigenmatrix[5,2,:] = -coeffAf*betaZ
    rightEigenmatrix[5,4,:] = rightEigenmatrix[5,2]
    rightEigenmatrix[5,5,:] = rightEigenmatrix[5,1]
    rightEigenmatrix[5,6,:] = rightEigenmatrix[5,0]

    #Modify some coefficients for the left eigenmatrix
    normFactor = 0.5/soundSpeedSq
    coeffQf = normFactor*coeffQf
    coeffQs = normFactor*coeffQs
    coeffAfPrime = normFactor*coeffAf*inverseDensity
    coeffAsPrime = normFactor*coeffAs*inverseDensity

    leftEigenmatrix[0,1,:] = -normFactor*fastMagSonSpeed*fastAlpha
    leftEigenmatrix[0,2,:] = coeffQs*betaY
    leftEigenmatrix[0,3,:] = coeffQs*betaZ
    leftEigenmatrix[0,6,:] = normFactor*fastAlpha*inverseDensity
    leftEigenmatrix[0,4,:] = coeffAsPrime*betaY
    leftEigenmatrix[0,5,:] = coeffAsPrime*betaZ

    leftEigenmatrix[1,2,:] = -0.5*betaZ
    leftEigenmatrix[1,3,:] = 0.5*betaY
    leftEigenmatrix[1,4,:] = -0.5*betaZ*BxSign/sqrtDens
    leftEigenmatrix[1,5,:] = 0.5*betaY*BxSign/sqrtDens

    leftEigenmatrix[2,1,:] = -1.0*normFactor*slowMagSonSpeed*slowAlpha
    leftEigenmatrix[2,2,:] = -1.0*coeffQf*betaY
    leftEigenmatrix[2,3,:] = -1.0*coeffQf*betaZ
    leftEigenmatrix[2,6,:] = normFactor*slowAlpha*inverseDensity
    leftEigenmatrix[2,4,:] = -1.0*coeffAfPrime*betaY
    leftEigenmatrix[2,5,:] = -1.0*coeffAfPrime*betaZ

    leftEigenmatrix[3,0,:] = 1.0
    leftEigenmatrix[3,6,:] = -1.0/soundSpeedSq

    leftEigenmatrix[4,1,:] = -1.0*leftEigenmatrix[2,1]
    leftEigenmatrix[4,2,:] = -1.0*leftEigenmatrix[2,2]
    leftEigenmatrix[4,3,:] = -1.0*leftEigenmatrix[2,3]
    leftEigenmatrix[4,6,:] = leftEigenmatrix[2,6]
    leftEigenmatrix[4,4,:] = leftEigenmatrix[2,4]
    leftEigenmatrix[4,5,:] = leftEigenmatrix[2,5]

    leftEigenmatrix[5,2,:] = -1.0*leftEigenmatrix[1,2]
    leftEigenmatrix[5,3,:] = -1.0*leftEigenmatrix[1,3]
    leftEigenmatrix[5,4,:] = leftEigenmatrix[1,4]
    leftEigenmatrix[5,5,:] = leftEigenmatrix[1,5]

    leftEigenmatrix[6,1,:] = -1.0*leftEigenmatrix[0,1]
    leftEigenmatrix[6,2,:] = -1.0*leftEigenmatrix[0,2]
    leftEigenmatrix[6,3,:] = -1.0*leftEigenmatrix[0,3]
    leftEigenmatrix[6,6,:] = leftEigenmatrix[0,6]
    leftEigenmatrix[6,4,:] = leftEigenmatrix[0,4]
    leftEigenmatrix[6,5,:] = leftEigenmatrix[0,5]

    # Return the eigenvalues, the left eigenmatrix, and the right eigenmatrix
    return (eigenVals, leftEigenmatrix, rightEigenmatrix)


#Function: getEigenmatricesY_mhd
#Purpose: Calculates the y-direction eigenvalues and eigenmatrices for every
#         cell in the simulation grid.
#Input Parameters: primVarsY (the primitive variables for every cell in the grid)
#                  By (the cell-centred y-component of the magnetic field for
#                      every cell in the simulation grid)
#                  gamma (the specific heat ratio for the ideal gas
#Outputs: eigenVals (the eigenvalues for every cell in the simulation grid)
#         leftEigenmatrix (the left eigenmatrix for every cell in the simulation grid)
#         rightEigenmatrix (the right eigenmatrix for every cell in the simulation grid)
def getEigenmatricesY_mhd(primVarsY, By, gamma):
    #Create matrices for the eigenvalues and eigenmatrices
    if len(primVarsY.shape) == 2: #if the sim is 1D
        eigenVals = np.zeros(shape=(7,primVarsY.shape[1]))
        leftEigenmatrix = np.zeros(shape=(7,7,primVarsY.shape[1]))
        rightEigenmatrix = np.zeros(shape=(7,7,primVarsY.shape[1]))
    elif len(primVarsY.shape) == 3: #if the sim is 2D
        eigenVals = np.zeros(shape=(7,primVarsY.shape[1],primVarsY.shape[2]))
        leftEigenmatrix = np.zeros(shape=(7,7,primVarsY.shape[1],primVarsY.shape[2]))
        rightEigenmatrix = np.zeros(shape=(7,7,primVarsY.shape[1],primVarsY.shape[2]))
    else: #if the sim is 3D
        eigenVals = np.zeros(shape=(7,primVarsY.shape[1],primVarsY.shape[2],primVarsY.shape[3]))
        leftEigenmatrix = np.zeros(shape=(7,7,primVarsY.shape[1],primVarsY.shape[2],primVarsY.shape[3]))
        rightEigenmatrix = np.zeros(shape=(7,7,primVarsY.shape[1],primVarsY.shape[2],primVarsY.shape[3]))

    #Calculate the transverse magnetic field intensity
    BtranSq = np.square(primVarsY[4]) + np.square(primVarsY[5])
    Btran = np.sqrt(BtranSq)
    inverseDens = 1.0/primVarsY[0]
    sqrtDens = np.sqrt(primVarsY[0])
    reducedBtranSq = BtranSq*inverseDens
    #Calculate the hydrodynamic sound speed
    gammaPresProd = gamma*primVarsY[6]
    soundSpeedSq = gammaPresProd*inverseDens
    soundSpeed = np.sqrt(soundSpeedSq)
    #Calculate the Alfven speed in the y-direction
    alfvenSpeedSq = By*By*inverseDens
    alfvenSpeed = np.sqrt(alfvenSpeedSq)

    #Calculate the fast magnetosonic speed
    speedSum = alfvenSpeedSq + reducedBtranSq + soundSpeedSq
    speedDiff = alfvenSpeedSq + reducedBtranSq - soundSpeedSq
    speedSqrt = np.sqrt((speedDiff*speedDiff
                         + 4.0*soundSpeedSq*reducedBtranSq).astype(np.double))
    fastMagSonSpeedSq = 0.5*(speedSum + speedSqrt)
    fastMagSonSpeed = np.sqrt(fastMagSonSpeedSq)

    #Calculate the slow magnetosonic speed
    slowMagSonSpeedSq = soundSpeedSq*alfvenSpeedSq/fastMagSonSpeedSq
    slowMagSonSpeed = np.sqrt(slowMagSonSpeedSq)

    #Calculate the BetaZ and BetaX coefficients
    betaZ = np.zeros(shape=Btran.shape)
    betaX = np.zeros(shape=Btran.shape)
    betaZ[Btran == 0.0] = 1.0
    betaZ[Btran > 0.0] = primVarsY[4][Btran > 0.0]/Btran[Btran > 0.0]
    betaX[Btran > 0.0] = primVarsY[5][Btran > 0.0]/Btran[Btran > 0.0]

    #Calculate the fastAlpha and slowAlpha coefficients
    fastAlpha = np.zeros(shape=betaZ.shape)
    slowAlpha = np.zeros(shape=betaZ.shape)
    fastAlphaOneIndices = speedSqrt == 0.0
    fastAlpha[fastAlphaOneIndices] = 1.0
    slowAlphaOneIndices = np.logical_and(np.logical_not(fastAlphaOneIndices), (soundSpeedSq - slowMagSonSpeedSq) <= 0.0)
    slowAlpha[slowAlphaOneIndices] = 1.0
    secondFastAlphaOneIndices = np.logical_and(np.logical_not(np.logical_or(fastAlphaOneIndices,slowAlphaOneIndices)),
                                               (fastMagSonSpeedSq-soundSpeedSq) <= 0.0)
    fastAlpha[secondFastAlphaOneIndices] = 1.0
    alphaCalcIndices = np.logical_not(np.logical_or(speedSqrt == 0,
                                                    np.logical_or(soundSpeedSq - slowMagSonSpeedSq <= 0.0,
                                                                  fastMagSonSpeedSq - soundSpeedSq <= 0.0)))
    fastAlpha[alphaCalcIndices] = np.sqrt((soundSpeedSq-slowMagSonSpeedSq)/speedSqrt)[alphaCalcIndices]
    slowAlpha[alphaCalcIndices] = np.sqrt((fastMagSonSpeedSq-soundSpeedSq)/speedSqrt)[alphaCalcIndices]

    #Get the sign of Bx in every cell
    BySign = np.sign(By)
    BySign[BySign == 0.0] = 1.0
    #Calculate additional coefficients that will simplify the eigenvector calculations
    coeffQf = fastMagSonSpeed*fastAlpha*BySign
    coeffQs = slowMagSonSpeed*slowAlpha*BySign
    coeffAf = soundSpeed*fastAlpha*sqrtDens
    coeffAs = soundSpeed*slowAlpha*sqrtDens

    #Calculate the seven wavespeed eigenvalues
    eigenVals[0,:] = primVarsY[2] - fastMagSonSpeed #vy - cf
    eigenVals[1,:] = primVarsY[2] - alfvenSpeed #vy - ca
    eigenVals[2,:] = primVarsY[2] - slowMagSonSpeed #vy - cs
    eigenVals[3,:] = primVarsY[2] #vy
    eigenVals[4,:] = primVarsY[2] + slowMagSonSpeed #vy + cs
    eigenVals[5,:] = primVarsY[2] + alfvenSpeed #vy + ca
    eigenVals[6,:] = primVarsY[2] + fastMagSonSpeed #vy + cf

    #Add values to the right eigenmatrix
    rightEigenmatrix[0,0,:] = primVarsY[0]*fastAlpha
    rightEigenmatrix[0,2,:] = primVarsY[0]*slowAlpha
    rightEigenmatrix[0,3,:] = 1.0
    rightEigenmatrix[0,4,:] = rightEigenmatrix[0,2]
    rightEigenmatrix[0,6,:] = rightEigenmatrix[0,0]

    rightEigenmatrix[2,0,:] = -1.0*fastMagSonSpeed*fastAlpha
    rightEigenmatrix[2,2,:] = -1.0*slowMagSonSpeed*slowAlpha
    rightEigenmatrix[2,4,:] = -1.0*rightEigenmatrix[2,2]
    rightEigenmatrix[2,6,:] = -1.0*rightEigenmatrix[2,0]

    rightEigenmatrix[3,0,:] = coeffQs*betaZ
    rightEigenmatrix[3,1,:] = -1.0*betaX
    rightEigenmatrix[3,2,:] = -1.0*coeffQf*betaZ
    rightEigenmatrix[3,4,:] = -1.0*rightEigenmatrix[3,2]
    rightEigenmatrix[3,5,:] = betaX
    rightEigenmatrix[3,6,:] = -1.0*rightEigenmatrix[3,0]

    rightEigenmatrix[1,0,:] = coeffQs*betaX
    rightEigenmatrix[1,1,:] = betaZ
    rightEigenmatrix[1,2,:] = -1.0*coeffQf*betaX
    rightEigenmatrix[1,4,:] = -1.0*rightEigenmatrix[1,2]
    rightEigenmatrix[1,5,:] = -1.0*betaZ
    rightEigenmatrix[1,6,:] = -1.0*rightEigenmatrix[1,0]

    rightEigenmatrix[6,0,:] = primVarsY[0]*soundSpeedSq*fastAlpha
    rightEigenmatrix[6,2,:] = primVarsY[0]*soundSpeedSq*slowAlpha
    rightEigenmatrix[6,4,:] = rightEigenmatrix[6,2]
    rightEigenmatrix[6,6,:] = rightEigenmatrix[6,0]

    rightEigenmatrix[4,0,:] = coeffAs*betaZ
    rightEigenmatrix[4,1,:] = -1.0*betaX*BySign*sqrtDens
    rightEigenmatrix[4,2,:] = -1.0*coeffAf*betaZ
    rightEigenmatrix[4,4,:] = rightEigenmatrix[4,2]
    rightEigenmatrix[4,5,:] = rightEigenmatrix[4,1]
    rightEigenmatrix[4,6,:] = rightEigenmatrix[4,0]

    rightEigenmatrix[5,0,:] = coeffAs*betaX
    rightEigenmatrix[5,1,:] = betaZ*BySign*sqrtDens
    rightEigenmatrix[5,2,:] = -1.0*coeffAf*betaX
    rightEigenmatrix[5,4,:] = rightEigenmatrix[5,2]
    rightEigenmatrix[5,5,:] = rightEigenmatrix[5,1]
    rightEigenmatrix[5,6,:] = rightEigenmatrix[5,0]

    #Modify some coefficients for the left eigenmatrix
    normFactor = 0.5/soundSpeedSq
    coeffQf = normFactor*coeffQf
    coeffQs = normFactor*coeffQs
    coeffAfPrime = normFactor*coeffAf*inverseDens
    coeffAsPrime = normFactor*coeffAs*inverseDens

    leftEigenmatrix[0,2,:] = -1.0*normFactor*fastMagSonSpeed*fastAlpha
    leftEigenmatrix[0,3,:] = coeffQs*betaZ
    leftEigenmatrix[0,1,:] = coeffQs*betaX
    leftEigenmatrix[0,6,:] = normFactor*fastAlpha*inverseDens
    leftEigenmatrix[0,4,:] = coeffAsPrime*betaZ
    leftEigenmatrix[0,5,:] = coeffAsPrime*betaX

    leftEigenmatrix[1,3,:] = -0.5*betaX
    leftEigenmatrix[1,1,:] = 0.5*betaZ
    leftEigenmatrix[1,4,:] = -0.5*betaX*BySign/sqrtDens
    leftEigenmatrix[1,5,:] = 0.5*betaZ*BySign/sqrtDens

    leftEigenmatrix[2,2,:] = -1.0*normFactor*slowMagSonSpeed*slowAlpha
    leftEigenmatrix[2,3,:] = -1.0*coeffQf*betaZ
    leftEigenmatrix[2,1,:] = -1.0*coeffQf*betaX
    leftEigenmatrix[2,6,:] = normFactor*slowAlpha*inverseDens
    leftEigenmatrix[2,4,:] = -1.0*coeffAfPrime*betaZ
    leftEigenmatrix[2,5,:] = -1.0*coeffAfPrime*betaX

    leftEigenmatrix[3,0,:] = 1.0
    leftEigenmatrix[3,6,:] = -1.0/soundSpeedSq

    leftEigenmatrix[4,1,:] = -1.0*leftEigenmatrix[2,1]
    leftEigenmatrix[4,2,:] = -1.0*leftEigenmatrix[2,2]
    leftEigenmatrix[4,3,:] = -1.0*leftEigenmatrix[2,3]
    leftEigenmatrix[4,4,:] = leftEigenmatrix[2,4]
    leftEigenmatrix[4,5,:] = leftEigenmatrix[2,5]
    leftEigenmatrix[4,6,:] = leftEigenmatrix[2,6]

    leftEigenmatrix[5,1,:] = -1.0*leftEigenmatrix[1,1]
    leftEigenmatrix[5,3,:] = -1.0*leftEigenmatrix[1,3]
    leftEigenmatrix[5,4,:] = leftEigenmatrix[1,4]
    leftEigenmatrix[5,5,:] = leftEigenmatrix[1,5]

    leftEigenmatrix[6,1,:] = -1.0*leftEigenmatrix[0,1]
    leftEigenmatrix[6,2,:] = -1.0*leftEigenmatrix[0,2]
    leftEigenmatrix[6,3,:] = -1.0*leftEigenmatrix[0,3]
    leftEigenmatrix[6,4,:] = leftEigenmatrix[0,4]
    leftEigenmatrix[6,5,:] = leftEigenmatrix[0,5]
    leftEigenmatrix[6,6,:] = leftEigenmatrix[0,6]

    # Return the eigenvalues, the left eigenmatrix, and the right eigenmatrix
    return (eigenVals, leftEigenmatrix, rightEigenmatrix)

#Function: getEigenmatricesZ_mhd
#Purpose: Calculates the z-direction eigenvalues and eigenmatrices for every
#         cell in the simulation grid.
#Input Parameters: primVarsXZ(the primitive variables for every cell in the grid)
#                  Bz (the cell-centred z-component of the magnetic field for
#                      every cell in the simulation grid)
#                  gamma (the specific heat ratio for the ideal gas
#Outputs: eigenVals (the eigenvalues for every cell in the simulation grid)
#         leftEigenmatrix (the left eigenmatrix for every cell in the simulation grid)
#         rightEigenmatrix (the right eigenmatrix for every cell in the simulation grid)
def getEigenmatricesZ_mhd(primVarsZ, Bz, gamma):
    #Create matrices for the eigenvalues and eigenmatrices
    if len(primVarsZ.shape) == 2: #if the sim is 1D
        eigenVals = np.zeros(shape=(7,primVarsZ.shape[1]))
        leftEigenmatrix = np.zeros(shape=(7,7,primVarsZ.shape[1]))
        rightEigenmatrix = np.zeros(shape=(7,7,primVarsZ.shape[1]))
    elif len(primVarsZ.shape) == 3: #if the sim is 2D
        eigenVals = np.zeros(shape=(7,primVarsZ.shape[1],primVarsZ.shape[2]))
        leftEigenmatrix = np.zeros(shape=(7,7,primVarsZ.shape[1],primVarsZ.shape[2]))
        rightEigenmatrix = np.zeros(shape=(7,7,primVarsZ.shape[1],primVarsZ.shape[2]))
    else: #if the sim is 3D
        eigenVals = np.zeros(shape=(7,primVarsZ.shape[1],primVarsZ.shape[2],primVarsZ.shape[3]))
        leftEigenmatrix = np.zeros(shape=(7,7,primVarsZ.shape[1],primVarsZ.shape[2],primVarsZ.shape[3]))
        rightEigenmatrix = np.zeros(shape=(7,7,primVarsZ.shape[1],primVarsZ.shape[2],primVarsZ.shape[3]))

    #Calculate the transverse magnetic field intensity
    BtranSq = np.square(primVarsZ[4]) + np.square(primVarsZ[5])
    Btran = np.sqrt(BtranSq)
    inverseDens = 1.0/primVarsZ[0]
    sqrtDens = np.sqrt(primVarsZ[0])
    gammaPresProd = gamma*primVarsZ[6]
    reducedBtranSq = BtranSq*inverseDens
    #Calculate the hydrodynamic sound speed
    soundSpeedSq = gammaPresProd*inverseDens
    soundSpeed = np.sqrt(soundSpeedSq)
    #Calculate the Alfven speed in the z-direction
    alfvenSpeedSq = Bz*Bz*inverseDens
    alfvenSpeed = np.sqrt(alfvenSpeedSq)

    #Calculate the fast magnetosonic speed
    speedSum = alfvenSpeedSq + reducedBtranSq + soundSpeedSq
    speedDiff = alfvenSpeedSq + reducedBtranSq - soundSpeedSq
    speedSqrt = np.sqrt((speedDiff*speedDiff
                         + 4.0*soundSpeedSq*reducedBtranSq).astype(np.double))
    fastMagSonSpeedSq = 0.5*(speedSum + speedSqrt)
    fastMagSonSpeed = np.sqrt(fastMagSonSpeedSq)

    #Calculate the slow magnetosonic speed
    slowMagSonSpeedSq = soundSpeedSq*alfvenSpeedSq/fastMagSonSpeedSq
    slowMagSonSpeed = np.sqrt(slowMagSonSpeedSq)


    #Calculate the BetaX and BetaY coefficients
    betaX = np.zeros(shape=Btran.shape)
    betaY = np.zeros(shape=Btran.shape)
    betaX[Btran == 0.0] = 1.0
    betaX[Btran > 0.0] = primVarsZ[4][Btran > 0.0]/Btran[Btran > 0.0]
    betaY[Btran > 0.0] = primVarsZ[5][Btran > 0.0]/Btran[Btran > 0.0]

    #Calculate the fastAlpha and slowAlpha coefficients
    fastAlpha = np.zeros(shape=betaX.shape)
    slowAlpha = np.zeros(shape=betaX.shape)
    fastAlphaOneIndices = speedSqrt == 0.0
    fastAlpha[fastAlphaOneIndices] = 1.0
    slowAlphaOneIndices = np.logical_and(np.logical_not(fastAlphaOneIndices), (soundSpeedSq - slowMagSonSpeedSq) <= 0.0)
    slowAlpha[slowAlphaOneIndices] = 1.0
    secondFastAlphaOneIndices = np.logical_and(np.logical_not(np.logical_or(fastAlphaOneIndices,slowAlphaOneIndices)),
                                               (fastMagSonSpeedSq-soundSpeedSq) <= 0.0)
    fastAlpha[secondFastAlphaOneIndices] = 1.0
    alphaCalcIndices = np.logical_not(np.logical_or(speedSqrt == 0,
                                                    np.logical_or(soundSpeedSq - slowMagSonSpeedSq <= 0.0,
                                                                  fastMagSonSpeedSq - soundSpeedSq <= 0.0)))
    fastAlpha[alphaCalcIndices] = np.sqrt((soundSpeedSq
                                           - slowMagSonSpeedSq)[alphaCalcIndices]
                                          / speedSqrt[alphaCalcIndices])
    slowAlpha[alphaCalcIndices] = np.sqrt((fastMagSonSpeedSq
                                           - soundSpeedSq)[alphaCalcIndices]
                                          / speedSqrt[alphaCalcIndices])

    #Get the sign of Bx in every cell
    BzSign = np.sign(Bz)
    BzSign[BzSign == 0.0] = 1.0
    #Calculate additional coefficients that will simplify the eigenvector calculations
    coeffQf = fastMagSonSpeed*fastAlpha*BzSign
    coeffQs = slowMagSonSpeed*slowAlpha*BzSign
    coeffAf = soundSpeed*fastAlpha*sqrtDens
    coeffAs = soundSpeed*slowAlpha*sqrtDens

    #Calculate the seven wavespeed eigenvalues
    eigenVals[0,:] = primVarsZ[3] - fastMagSonSpeed #vz - cf
    eigenVals[1,:] = primVarsZ[3] - alfvenSpeed #vz - ca
    eigenVals[2,:] = primVarsZ[3] - slowMagSonSpeed #vz - cs
    eigenVals[3,:] = primVarsZ[3] #vz
    eigenVals[4,:] = primVarsZ[3] + slowMagSonSpeed #vz + cs
    eigenVals[5,:] = primVarsZ[3] + alfvenSpeed #vz + ca
    eigenVals[6,:] = primVarsZ[3] + fastMagSonSpeed #vz + cf

    #Add values to the right eigenmatrix
    rightEigenmatrix[0,0,:] = primVarsZ[0]*fastAlpha
    rightEigenmatrix[0,2,:] = primVarsZ[0]*slowAlpha
    rightEigenmatrix[0,3,:] = 1.0
    rightEigenmatrix[0,4,:] = rightEigenmatrix[0,2]
    rightEigenmatrix[0,6,:] = rightEigenmatrix[0,0]

    rightEigenmatrix[3,0,:] = -1.0*fastMagSonSpeed*fastAlpha
    rightEigenmatrix[3,2,:] = -1.0*slowMagSonSpeed*slowAlpha
    rightEigenmatrix[3,4,:] = -1.0*rightEigenmatrix[3,2]
    rightEigenmatrix[3,6,:] = -1.0*rightEigenmatrix[3,0]

    rightEigenmatrix[1,0,:] = coeffQs*betaX
    rightEigenmatrix[1,1,:] = -1.0*betaY
    rightEigenmatrix[1,2,:] = -1.0*coeffQf*betaX
    rightEigenmatrix[1,4,:] = -1.0*rightEigenmatrix[1,2]
    rightEigenmatrix[1,5,:] = betaY
    rightEigenmatrix[1,6,:] = -1.0*rightEigenmatrix[1,0]

    rightEigenmatrix[2,0,:] = coeffQs*betaY
    rightEigenmatrix[2,1,:] = betaX
    rightEigenmatrix[2,2,:] = -1.0*coeffQf*betaY
    rightEigenmatrix[2,4,:] = -1.0*rightEigenmatrix[2,2]
    rightEigenmatrix[2,5,:] = -1.0*betaX
    rightEigenmatrix[2,6,:] = -1.0*rightEigenmatrix[2,0]

    rightEigenmatrix[6,0,:] = primVarsZ[0]*soundSpeedSq*fastAlpha
    rightEigenmatrix[6,2,:] = primVarsZ[0]*soundSpeedSq*slowAlpha
    rightEigenmatrix[6,4,:] = rightEigenmatrix[6,2]
    rightEigenmatrix[6,6,:] = rightEigenmatrix[6,0]

    rightEigenmatrix[4,0,:] = coeffAs*betaX
    rightEigenmatrix[4,1,:] = -1.0*betaY*BzSign*sqrtDens
    rightEigenmatrix[4,2,:] = -1.0*coeffAf*betaX
    rightEigenmatrix[4,4,:] = rightEigenmatrix[4,2]
    rightEigenmatrix[4,5,:] = rightEigenmatrix[4,1]
    rightEigenmatrix[4,6,:] = rightEigenmatrix[4,0]

    rightEigenmatrix[5,0,:] = coeffAs*betaY
    rightEigenmatrix[5,1,:] = betaX*BzSign*sqrtDens
    rightEigenmatrix[5,2,:] = -1.0*coeffAf*betaY
    rightEigenmatrix[5,4,:] = rightEigenmatrix[5,2]
    rightEigenmatrix[5,5,:] = rightEigenmatrix[5,1]
    rightEigenmatrix[5,6,:] = rightEigenmatrix[5,0]

    #Modify some coefficients for the left eigenmatrix
    normFactor = 0.5/soundSpeedSq
    coeffQf = normFactor*coeffQf
    coeffQs = normFactor*coeffQs
    coeffAfPrime = normFactor*coeffAf*inverseDens
    coeffAsPrime = normFactor*coeffAs*inverseDens

    leftEigenmatrix[0,3,:] = -1.0*normFactor*fastMagSonSpeed*fastAlpha
    leftEigenmatrix[0,1,:] = coeffQs*betaX
    leftEigenmatrix[0,2,:] = coeffQs*betaY
    leftEigenmatrix[0,6,:] = normFactor*fastAlpha*inverseDens
    leftEigenmatrix[0,4,:] = coeffAsPrime*betaX
    leftEigenmatrix[0,5,:] = coeffAsPrime*betaY

    leftEigenmatrix[1,1,:] = -0.5*betaY
    leftEigenmatrix[1,2,:] = 0.5*betaX
    leftEigenmatrix[1,4,:] = -0.5*betaY*BzSign/sqrtDens
    leftEigenmatrix[1,5,:] = 0.5*betaX*BzSign/sqrtDens

    leftEigenmatrix[2,3,:] = -1.0*normFactor*slowMagSonSpeed*slowAlpha
    leftEigenmatrix[2,1,:] = -1.0*coeffQf*betaX
    leftEigenmatrix[2,2,:] = -1.0*coeffQf*betaY
    leftEigenmatrix[2,6,:] = normFactor*slowAlpha*inverseDens
    leftEigenmatrix[2,4,:] = -1.0*coeffAfPrime*betaX
    leftEigenmatrix[2,5,:] = -1.0*coeffAfPrime*betaY

    leftEigenmatrix[3,0,:] = 1.0
    leftEigenmatrix[3,6,:] = -1.0/soundSpeedSq

    leftEigenmatrix[4,1,:] = -1.0*leftEigenmatrix[2,1]
    leftEigenmatrix[4,2,:] = -1.0*leftEigenmatrix[2,2]
    leftEigenmatrix[4,3,:] = -1.0*leftEigenmatrix[2,3]
    leftEigenmatrix[4,4,:] = leftEigenmatrix[2,4]
    leftEigenmatrix[4,5,:] = leftEigenmatrix[2,5]
    leftEigenmatrix[4,6,:] = leftEigenmatrix[2,6]

    leftEigenmatrix[5,1,:] = -1.0*leftEigenmatrix[1,1]
    leftEigenmatrix[5,2,:] = -1.0*leftEigenmatrix[1,2]
    leftEigenmatrix[5,4,:] = leftEigenmatrix[1,4]
    leftEigenmatrix[5,5,:] = leftEigenmatrix[1,5]

    leftEigenmatrix[6,1,:] = -1.0*leftEigenmatrix[0,1]
    leftEigenmatrix[6,2,:] = -1.0*leftEigenmatrix[0,2]
    leftEigenmatrix[6,3,:] = -1.0*leftEigenmatrix[0,3]
    leftEigenmatrix[6,4,:] = leftEigenmatrix[0,4]
    leftEigenmatrix[6,5,:] = leftEigenmatrix[0,5]
    leftEigenmatrix[6,6,:] = leftEigenmatrix[0,6]

    # Return the eigenvalues, the left eigenmatrix, and the right eigenmatrix
    return (eigenVals, leftEigenmatrix, rightEigenmatrix)