#CT_HelperModule.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides functions for deriving cell-centred magnetic fields from face-centred
#         magnetic field values and vice versa, in accordance with the Constrained Transport
#         method described in (Evans & Hawley, 1988) [1] and (Stone et al., 2008) [2].
#References:
# 1. Evans, C. R., & Hawley, J. F. (1988). Simulation of magnetohydrodynamic flows: A Constrained Transport method.
#    Astrophysical Journal, 332, 659-677. https://adsabs.harvard.edu/full/1988ApJ...332..659E
# 2. Stone, J. M., Gardiner, T. A., Teuben, P., Hawley, J. F., & Simon, J. B. (2008).
#    Athena: A new code for astrophysical MHD. The Astrophysical Journal Supplemental Series,
#    178(1), 137-177. https://iopscience.iop.org/article/10.1086/588755/pdf

######IMPORT STATEMENTS######

#Import NumPy for matrix operations
import numpy as np

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants

#####CELL-CENTRED -> FACE-CENTRED CONVERSIONS (X-DIRECTION)######
#Function: getFaceBx_2D
#Purpose: Calculates the initial face-centred Bx values for a 2D simulation if they have not been specified by the user.
#Input Parameters: centBx (the cell-centred Bx values for the 2D simulation)
#                  BcX (integer for the boundary condition in the x-direction
#                      (0 = outflow and 1 = periodic))
#Outputs: faceBx (a matrix that contains the face-centred Bx value for each intercell boundary in the x-direction)
def getFaceBx_2D(centBx,BcX):
    if BcX == constants.OUTFLOW:
        leftCentBx = np.append(centBx[:,0].reshape(-1,1),centBx,axis=1)
        rightCentBx = np.append(centBx,centBx[:,centBx.shape[1]-1].reshape(-1,1),axis=1)
    else:
        leftCentBx = np.append(centBx[:,centBx.shape[1]-1].reshape(-1,1),centBx,axis=1)
        rightCentBx = np.append(centBx,centBx[:,0].reshape(-1,1),axis=1)
    faceBx = (leftCentBx + rightCentBx)/2.0
    return faceBx

#Function: getFaceBx_3D
#Purpose: Calculates the initial face-centred Bx values for a 3D simulation if they have not been specified by the user.
#Input Parameters: centBx (the cell-centred Bx values for the 3D simulation)
#                  BcX (integer for the boundary condition in the x-direction
#                      (0 = outflow and 1 = periodic))
#Outputs: faceBx (a matrix that contains the face-centred Bx value for each intercell boundary in the x-direction)
def getFaceBx_3D(centBx,BcX):
    if BcX == constants.OUTFLOW:
        leftCentBx = np.append(centBx[:,0,:].reshape(centBx.shape[0],1,-1),centBx,axis=1)
        rightCentBx = np.append(centBx,centBx[:,centBx.shape[1]-1,:].reshape(centBx.shape[0],1,-1),axis=1)
    else:
        leftCentBx = np.append(centBx[:,centBx.shape[1]-1,:].reshape(centBx.shape[0],1,-1),centBx,axis=1)
        rightCentBx = np.append(centBx,centBx[:,0,:].reshape(centBx.shape[0],1,-1),axis=1)
    faceBx = 0.5*(leftCentBx + rightCentBx)
    return faceBx

#####CELL-CENTRED -> FACE-CENTRED CONVERSIONS (Y-DIRECTION)######
#Function: getFaceBy_2D
#Purpose: Calculates the initial face-centred By values for a 2D simulation if they have not been specified by the user.
#Input Parameters: centBy (the cell-centred By values for the 2D simulation)
#                  BcY (integer for the boundary condition in the y-direction
#                      (0 = outflow and 1 = periodic))
#Outputs: faceBy (a matrix that contains the face-centred Bx value for each intercell boundary in the y-direction)
def getFaceBy_2D(centBy,BcY):
    if BcY == constants.OUTFLOW:
        topCentBy = np.append(centBy[0,:].reshape(1,-1),centBy,axis=0)
        bottomCentBy = np.append(centBy,centBy[centBy.shape[0]-1,:].reshape(1,-1),axis=0)
    else:
        topCentBy = np.append(centBy[centBy.shape[0]-1,:].reshape(1,-1),centBy,axis=0)
        bottomCentBy = np.append(centBy,centBy[0,:].reshape(1,-1),axis=0)
    faceBy = (topCentBy + bottomCentBy)/2.0
    return faceBy

#Function: getFaceBy_3D
#Purpose: Calculates the initial face-centred By values for a 3D simulation if they have not been specified by the user.
#Input Parameters: centBy (the cell-centred By values for the 3D simulation)
#                  BcY (integer for the boundary condition in the y-direction
#                      (0 = outflow and 1 = periodic))
#Outputs: faceBy (a matrix that contains the face-centred By value for each intercell boundary in the y-direction)
def getFaceBy_3D(centBy,BcY):
    if BcY == constants.OUTFLOW:
        topCentBy = np.append(centBy[0,:,:].reshape(1,centBy.shape[1],-1),centBy,axis=0)
        bottomCentBy = np.append(centBy,centBy[centBy.shape[0]-1,:,:].reshape(1,centBy.shape[1],-1),axis=0)
    else:
        topCentBy = np.append(centBy[centBy.shape[0]-1,:,:].reshape(1,centBy.shape[1],-1),centBy,axis=0)
        bottomCentBy = np.append(centBy,centBy[0,:,:].reshape(1,centBy.shape[1],-1),axis=0)
    faceBy = (topCentBy + bottomCentBy)/2.0
    return faceBy

#####CELL-CENTRED -> FACE-CENTRED CONVERSION (Z-DIRECTION)######
#Function: getFaceBz_3D
#Purpose: Calculates the initial face-centred Bz values for a 3D simulation if they have not been specified by the user.
#Input Parameters: centBz (the cell-centred Bz values for the 3D simulation)
#                  BcZ (integer for the boundary condition in the z-direction
#                      (0 = outflow and 1 = periodic))
#Outputs: faceBz (a matrix that contains the face-centred Bz value for each intercell boundary in the z-direction)
def getFaceBz_3D(centBz,BcZ):
    if BcZ == constants.OUTFLOW:
        backCentBz = np.append(centBz[:,:,0].reshape(centBz.shape[0],-1,1),centBz,axis=2)
        forwCentBz = np.append(centBz,centBz[:,:,centBz.shape[2]-1].reshape(centBz.shape[0],-1,1),axis=2)
    else:
        backCentBz = np.append(centBz[:,:,centBz.shape[2]-1].reshape(centBz.shape[0],-1,1),centBz,axis=2)
        forwCentBz = np.append(centBz,centBz[:,:,0].reshape(centBz.shape[0],-1,1),axis=2)
    faceBz = (backCentBz + forwCentBz)/2.0
    return faceBz

#####FACE-CENTRED -> CELL-CENTRED CONVERSIONS (X-DIRECTION)######
#Function: getCentBx
#Purpose: Calculates the cell-centred Bx values from the face-centred Bx values in a 2D or 3D simulation.
#Input Parameters: faceBx (the face-centred Bx values)
#Outputs: centBx (a matrix that contains the cell-centred Bx value for each cell in the simulation grid)
def getCentBx(faceBx):
    leftBx = faceBx[:,0:faceBx.shape[1]-1]
    rightBx = faceBx[:,1:faceBx.shape[1]]
    centBx = (leftBx + rightBx)/2.0
    return centBx

#####FACE-CENTRED -> CELL-CENTRED CONVERSIONS (Y-DIRECTION)######
#Function: getCentBy
#Purpose: Calculates the cell-centred By values from the face-centred By values in a 2D or 3D simulation.
#Input Parameters: faceBy (the face-centred Bx values)
#Outputs: centBy (a matrix that contains the cell-centred By value for each cell in the simulation grid)
def getCentBy(faceBy):
    topBy = faceBy[0:faceBy.shape[0]-1,:]
    bottomBy = faceBy[1:faceBy.shape[0],:]
    centBy = (topBy + bottomBy)/2.0
    return centBy

#####FACE-CENTRED -> CELL-CENTRED CONVERSIONS (Z-DIRECTION)######
#Function: getCentBz
#Purpose: Calculates the cell-centred Bx values from the face-centred Bz values in a 2D or 3D simulation.
#Input Parameters: faceBz (the face-centred Bz values)
#Outputs: centBz (a matrix that contains the cell-centred Bz value for each cell in the simulation grid)
def getCentBz(faceBz):
    backBz = faceBz[:,:,0:faceBz.shape[2]-1]
    forwBz = faceBz[:,:,1:faceBz.shape[2]]
    centBz = (backBz + forwBz)/2.0
    return centBz
