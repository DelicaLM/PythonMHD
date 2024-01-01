#PrimCons_HelperModule.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides functions for converting between primitive and conservative variables
#         (used by the Simulation Grid and Simulation classes in Simulation.py,
#          as well as most of the helper modules).

######IMPORT STATEMENTS######

#Import Numpy for matrix operations
import numpy as np

#Import PythonMHD constants
import Source.PythonMHD_Constants as constants

#Function: primToCons_hydro
#Purpose: Converts primitive variables (density, velocities, pressure) into conservative variables
#         (density, momenta, energy) for hydrodynamic simulations, and returns the result as a new matrix.
#         This function assumes that the gas is ideal with an adiabatic equation of state.
#Input Parameters: primVars (the primitive variables matrix)
#                  gamma (the specific heat ratio of the ideal gas)
#Outputs: consVars (the conservative variables matrix)
def primToCons_hydro(primVars, gamma):
    consVars = np.zeros(shape=primVars.shape)
    #Density is left unchanged
    consVars[0,:] = primVars[0,:]
    #Convert the velocities to momenta
    consVars[1,:] = primVars[0,:]*primVars[1,:]
    consVars[2,:] = primVars[0,:]*primVars[2,:]
    consVars[3,:] = primVars[0,:]*primVars[3,:]
    #Convert the hydrodynamic pressure to total energy
    #(E = P/(gamma-1) + 0.5*rho*V^2)
    consVars[4,:] = (primVars[4,:]/(gamma-1.0)) \
                     + 0.5*primVars[0,:]*(np.square(primVars[1,:]) + np.square(primVars[2,:])
                                           + np.square(primVars[3,:]))
    #Return the conservative variables
    return consVars

#Function: primToCons_hydro
#Purpose: Converts primitive variables (density, velocities, pressure) into conservative variables
#         (density, momenta, energy) for hydrodynamic simulations, and returns the result as a new matrix.
#         This function assumes that the gas is ideal with an adiabatic equation of state.
#Input Parameters: primVars (the primitive variables matrix)
#                  gamma (the specific heat ratio of the ideal gas)
#Outputs: consVars (the conservative variables matrix)
def primToCons_hydro_x(primVars, gamma):
    consVars = np.zeros(shape=primVars.shape)
    #Density is left unchanged
    consVars[0,:] = primVars[0,:]
    #Convert the velocities to momenta
    consVars[1,:] = primVars[0,:]*primVars[1,:]
    consVars[2,:] = primVars[0,:]*primVars[2,:]
    consVars[3,:] = primVars[0,:]*primVars[3,:]
    #Convert the hydrodynamic pressure to total energy
    #(E = P/(gamma-1) + 0.5*rho*V^2)
    consVars[4,:] = (primVars[4,:]/(gamma-1.0)) \
                     + 0.5*primVars[0,:]*(np.square(primVars[1,:]) + np.square(primVars[2,:])
                                           + np.square(primVars[3,:]))
    #Return the conservative variables
    return consVars

#Function: primToCons_hydro
#Purpose: Converts primitive variables (density, velocities, pressure) into conservative variables
#         (density, momenta, energy) for hydrodynamic simulations, and returns the result as a new matrix.
#         This function assumes that the gas is ideal with an adiabatic equation of state.
#Input Parameters: primVars (the primitive variables matrix)
#                  gamma (the specific heat ratio of the ideal gas)
#Outputs: consVars (the conservative variables matrix)
def primToCons_hydro_y(primVars, gamma):
    consVars = np.zeros(shape=primVars.shape)
    #Density is left unchanged
    consVars[0,:] = primVars[0,:]
    #Convert the velocities to momenta
    consVars[1,:] = primVars[0,:]*primVars[1,:]
    consVars[2,:] = primVars[0,:]*primVars[2,:]
    consVars[3,:] = primVars[0,:]*primVars[3,:]
    #Convert the hydrodynamic pressure to total energy
    #(E = P/(gamma-1) + 0.5*rho*V^2)
    consVars[4,:] = (primVars[4,:]/(gamma-1.0)) \
                     + 0.5*primVars[0,:]*(np.square(primVars[2,:]) + np.square(primVars[3,:])
                                           + np.square(primVars[1,:]))
    #Return the conservative variables
    return consVars

#Function: primToCons_hydro
#Purpose: Converts primitive variables (density, velocities, pressure) into conservative variables
#         (density, momenta, energy) for hydrodynamic simulations, and returns the result as a new matrix.
#         This function assumes that the gas is ideal with an adiabatic equation of state.
#Input Parameters: primVars (the primitive variables matrix)
#                  gamma (the specific heat ratio of the ideal gas)
#Outputs: consVars (the conservative variables matrix)
def primToCons_hydro_z(primVars, gamma):
    consVars = np.zeros(shape=primVars.shape)
    #Density is left unchanged
    consVars[0,:] = primVars[0,:]
    #Convert the velocities to momenta
    consVars[1,:] = primVars[0,:]*primVars[1,:]
    consVars[2,:] = primVars[0,:]*primVars[2,:]
    consVars[3,:] = primVars[0,:]*primVars[3,:]
    #Convert the hydrodynamic pressure to total energy
    #(E = P/(gamma-1) + 0.5*rho*V^2)
    consVars[4,:] = (primVars[4,:]/(gamma-1.0)) \
                     + 0.5*primVars[0,:]*(np.square(primVars[3,:]) + np.square(primVars[1,:])
                                           + np.square(primVars[2,:]))
    #Return the conservative variables
    return consVars

#Function: consToPrim_hydro
#Purpose: Converts conservative variables (density, momenta, energy) into primitive variables
#         (density, velocities, pressure) for hydrodynamic simulations, and returns the result
#         as a new matrix. This function assumes that the gas is ideal with an adiabatic equation
#         of state.
#Input Parameters: consVars (the conservative variables matrix)
#                  gamma (the specific heat ratio of the ideal gas)
#Outputs: consVars (the conservative variables matrix)
def consToPrim_hydro(consVars,gamma):
    primVars = np.zeros(shape=consVars.shape)
    #Density is left unchanged
    primVars[0,:] = consVars[0,:]
    inverseDens = 1.0/primVars[0,:]
    #Convert momenta to velocities
    primVars[1,:] = consVars[1,:]*inverseDens
    primVars[2,:] = consVars[2,:]*inverseDens
    primVars[3,:] = consVars[3,:]*inverseDens
    #Convert total energy to hydrodynamic pressure
    #(P = (gamma-1)*(E - 0.5*rho*V^2)
    primVars[4,:] = consVars[4,:] - 0.5*(np.square(consVars[1,:]) + np.square(consVars[2,:])
                                         + np.square(consVars[3,:]))*inverseDens
    primVars[4,:] *= (gamma-1.0)
    #Check if pressure is negative and correct (figure out how to let user know
    #this has occurred)
    #Return the primitive variables
    return primVars

def consToPrim_hydro_x(consVars,gamma):
    primVars = np.zeros(shape=consVars.shape)
    #Density is left unchanged
    primVars[0,:] = consVars[0,:]
    inverseDens = 1.0/primVars[0,:]
    #Convert momenta to velocities
    primVars[1,:] = consVars[1,:]*inverseDens
    primVars[2,:] = consVars[2,:]*inverseDens
    primVars[3,:] = consVars[3,:]*inverseDens
    #Convert total energy to hydrodynamic pressure
    #(P = (gamma-1)*(E - 0.5*rho*V^2)
    primVars[4,:] = consVars[4,:] - 0.5*(np.square(consVars[1,:]) + np.square(consVars[2,:])
                                         + np.square(consVars[3,:]))*inverseDens
    primVars[4,:] *= (gamma-1.0)
    #Check if pressure is negative and correct (figure out how to let user know
    #this has occurred)
    #Return the primitive variables
    return primVars

def consToPrim_hydro_y(consVars,gamma):
    primVars = np.zeros(shape=consVars.shape)
    #Density is left unchanged
    primVars[0,:] = consVars[0,:]
    inverseDens = 1.0/primVars[0,:]
    #Convert momenta to velocities
    primVars[1,:] = consVars[1,:]*inverseDens
    primVars[2,:] = consVars[2,:]*inverseDens
    primVars[3,:] = consVars[3,:]*inverseDens
    #Convert total energy to hydrodynamic pressure
    #(P = (gamma-1)*(E - 0.5*rho*V^2)
    primVars[4,:] = consVars[4,:] - 0.5*(np.square(consVars[2,:]) + np.square(consVars[3,:])
                                         + np.square(consVars[1,:]))*inverseDens
    primVars[4,:] *= (gamma-1.0)
    #Check if pressure is negative and correct (figure out how to let user know
    #this has occurred)
    #Return the primitive variables
    return primVars

def consToPrim_hydro_z(consVars,gamma):
    primVars = np.zeros(shape=consVars.shape)
    #Density is left unchanged
    primVars[0,:] = consVars[0,:]
    inverseDens = 1.0/primVars[0,:]
    #Convert momenta to velocities
    primVars[1,:] = consVars[1,:]*inverseDens
    primVars[2,:] = consVars[2,:]*inverseDens
    primVars[3,:] = consVars[3,:]*inverseDens
    #Convert total energy to hydrodynamic pressure
    #(P = (gamma-1)*(E - 0.5*rho*V^2)
    primVars[4,:] = consVars[4,:] - 0.5*(np.square(consVars[3,:]) + np.square(consVars[1,:])
                                         + np.square(consVars[2,:]))*inverseDens
    primVars[4,:] *= (gamma-1.0)
    #Check if pressure is negative and correct (figure out how to let user know
    #this has occurred)
    #Return the primitive variables
    return primVars

#Function: primToCons_mhd
#Purpose: Converts primitive variables (density, velocities, magnetic field components, pressure) into
#         conservative variables (density, momenta, magnetic field components, energy) for MHD simulations,
#         and returns the result as a new matrix. This function assumes that the gas is ideal with an
#         adiabatic equation of state.
#         Because we have different primitive variable vectors for each direction in MHD simulations
#         (primVarsX = (density, vx, vy, vz, By, Bz, pres), primVarsY = (density, vx, vy, vz, Bz, Bx, pres),
#          primVarsZ = (density, vx, vy, vz, Bx, By, pres)) we require an additional parameter for the
#          magnetic field component that is along the same dimension as the primitive variables vector
#          (i.e., the "in plane" component of the magnetic field).
#Input Parameters: primVars (the primitive variables matrix)
#                  inPlaneB (the in-plane component of the magnetic field
#                            (Bx for primVarsX, By for primVarsY, Bz for primVarsZ))
#                  gamma (the specific heat ratio of the ideal gas)
#Outputs: consVars (the conservative variables matrix, which also will not include
#                   the in-plane magnetic field component)
def primToCons_mhd(primVars, inPlaneB, gamma):
    #Create a matrix for the conservative variables
    consVars = np.zeros(shape=primVars.shape)
    #Density is left unchanged
    consVars[0,:] = primVars[0,:]
    #Convert velocities to momenta
    consVars[1,:] = primVars[0,:]*primVars[1,:]
    consVars[2,:] = primVars[0,:]*primVars[2,:]
    consVars[3,:] = primVars[0,:]*primVars[3,:]
    #Magnetic field components are left unchanged
    consVars[4,:] = primVars[4,:]
    consVars[5,:] = primVars[5,:]
    #Convert hydrodynamic pressure to total energy
    #(E = P/(gamma-1) + 0.5*rho*V^2 + 0.5*B^2 (with permeability of free space u0 set to 1))
    consVars[6,:] = primVars[6,:]/(gamma-1.0) \
                        + 0.5*primVars[0,:]*(primVars[1,:]*primVars[1,:]
                                             + primVars[2,:]*primVars[2,:]
                                             + primVars[3,:]*primVars[3,:])
    consVars[6] += 0.5*(inPlaneB*inPlaneB + primVars[4,:]*primVars[4,:]
                                 + primVars[5,:]*primVars[5,:])
    #Return the conservative variables
    return consVars


def primToCons_mhd_x(primVars, inPlaneB, gamma):
    #Create a matrix for the conservative variables
    consVars = np.zeros(shape=primVars.shape)
    #Density is left unchanged
    consVars[0,:] = primVars[0,:]
    #Convert velocities to momenta
    consVars[1,:] = primVars[0,:]*primVars[1,:]
    consVars[2,:] = primVars[0,:]*primVars[2,:]
    consVars[3,:] = primVars[0,:]*primVars[3,:]
    #Magnetic field components are left unchanged
    consVars[4,:] = primVars[4,:]
    consVars[5,:] = primVars[5,:]
    #Convert hydrodynamic pressure to total energy
    #(E = P/(gamma-1) + 0.5*rho*V^2 + 0.5*B^2 (with permeability of free space u0 set to 1))
    consVars[6,:] = primVars[6,:]/(gamma-1.0) \
                        + 0.5*primVars[0,:]*(primVars[1,:]*primVars[1,:]
                                             + primVars[2,:]*primVars[2,:]
                                             + primVars[3,:]*primVars[3,:])
    consVars[6] += 0.5*(inPlaneB*inPlaneB + primVars[4,:]*primVars[4,:]
                                 + primVars[5,:]*primVars[5,:])
    #Return the conservative variables
    return consVars

def primToCons_mhd_y(primVars, inPlaneB, gamma):
    #Create a matrix for the conservative variables
    consVars = np.zeros(shape=primVars.shape)
    #Density is left unchanged
    consVars[0,:] = primVars[0,:]
    #Convert velocities to momenta
    consVars[1,:] = primVars[0,:]*primVars[1,:]
    consVars[2,:] = primVars[0,:]*primVars[2,:]
    consVars[3,:] = primVars[0,:]*primVars[3,:]
    #Magnetic field components are left unchanged
    consVars[4,:] = primVars[4,:]
    consVars[5,:] = primVars[5,:]
    #Convert hydrodynamic pressure to total energy
    #(E = P/(gamma-1) + 0.5*rho*V^2 + 0.5*B^2 (with permeability of free space u0 set to 1))
    consVars[6,:] = primVars[6,:]/(gamma-1.0) \
                        + 0.5*primVars[0,:]*(primVars[2,:]*primVars[2,:]
                                             + primVars[3,:]*primVars[3,:]
                                             + primVars[1,:]*primVars[1,:])
    consVars[6] += 0.5*(inPlaneB*inPlaneB + primVars[4,:]*primVars[4,:]
                                 + primVars[5,:]*primVars[5,:])
    #Return the conservative variables
    return consVars

def primToCons_mhd_z(primVars, inPlaneB, gamma):
    #Create a matrix for the conservative variables
    consVars = np.zeros(shape=primVars.shape)
    #Density is left unchanged
    consVars[0,:] = primVars[0,:]
    #Convert velocities to momenta
    consVars[1,:] = primVars[0,:]*primVars[1,:]
    consVars[2,:] = primVars[0,:]*primVars[2,:]
    consVars[3,:] = primVars[0,:]*primVars[3,:]
    #Magnetic field components are left unchanged
    consVars[4,:] = primVars[4,:]
    consVars[5,:] = primVars[5,:]
    #Convert hydrodynamic pressure to total energy
    #(E = P/(gamma-1) + 0.5*rho*V^2 + 0.5*B^2 (with permeability of free space u0 set to 1))
    consVars[6,:] = primVars[6,:]/(gamma-1.0) \
                        + 0.5*primVars[0,:]*(primVars[3,:]*primVars[3,:]
                                             + primVars[1,:]*primVars[1,:]
                                             + primVars[2,:]*primVars[2,:])
    consVars[6] += 0.5*(inPlaneB*inPlaneB + primVars[4,:]*primVars[4,:]
                                 + primVars[5,:]*primVars[5,:])
    #Return the conservative variables
    return consVars

#Function: primToCons_mhd_allVars
#Purpose: Converts primitive variables (density, velocities, magnetic field components, pressure) into
#         conservative variables (density, momenta, magnetic field components, energy) for MHD simulations,
#         and returns the result as a new matrix. This function assumes that the gas is ideal with an
#         adiabatic equation of state.
#         Unlike primToCons_mhd, primToCons_mhd_allVars assumes that all magnetic field components
#         are included in the primitive variables matrix (ordered as {density, vx, vy, vz, Bx, By, Bz,
#         pressure}).
#Input Parameters: primVarsAll (the primitive variables matrix, which contains all eight
#                               primitive variables for MHD simulations)
#                  gamma (the specific heat ratio of the ideal gas)
#Outputs: consVarsAll (the conservative variables matrix, which will include all three magnetic field
#                                                         components)
def primToCons_mhd_allVars(primVarsAll, gamma):
    assert(primVarsAll.shape[0] == 8)
    #Create a matrix for the conservative variables
    consVarsAll = np.zeros(shape=primVarsAll.shape)
    #Density is left unchanged
    consVarsAll[0,:] = primVarsAll[0,:]
    #Convert velocities to momenta
    consVarsAll[1,:] = primVarsAll[0,:]*primVarsAll[1,:]
    consVarsAll[2,:] = primVarsAll[0,:]*primVarsAll[2,:]
    consVarsAll[3,:] = primVarsAll[0,:]*primVarsAll[3,:]
    #Magnetic field components are left unchanged
    consVarsAll[4,:] = primVarsAll[4,:]
    consVarsAll[5,:] = primVarsAll[5,:]
    consVarsAll[6,:] = primVarsAll[6,:]
    #Convert hydrodynamic pressure to total energy
    #(E = P/(gamma-1) + 0.5*rho*V^2 + 0.5*B^2 (with permeability of free space u0 set to 1))
    consVarsAll[7,:] = primVarsAll[7,:]/(gamma-1.0) \
                       + 0.5*primVarsAll[0]*(primVarsAll[1]*primVarsAll[1]
                                             + primVarsAll[2]*primVarsAll[2]
                                             + primVarsAll[3]*primVarsAll[3])

    consVarsAll[7] += 0.5*(primVarsAll[4]*primVarsAll[4]
                                +primVarsAll[5]*primVarsAll[5]
                                +primVarsAll[6]*primVarsAll[6])

    #Return the conservative variables
    return consVarsAll

#Function: consToPrim_mhd
#Purpose: Converts conservative variables (density, momenta, magnetic field components, energy) into
#         primitive variables (density, velocities, magnetic field components, pressure) for MHD simulations,
#         and returns the result as a new matrix. This function assumes that the gas is ideal with an
#         adiabatic equation of state.
#         Because we have different conservative variable vectors for each direction in MHD simulations
#         (consVarsX = (density, Mx, My, Mz, By, Bz, energy), primVarsY = (density, Mx, My, Mz, Bz, Bx, energy),
#          consVarsZ = (density, Mx, My, Mz, Bx, By, energy)) we require an additional parameter for the
#          magnetic field component that is along the same dimension as the conservative variables vector
#          (i.e., the "in plane" component of the magnetic field).
#Input Parameters: consVars (the conservative variables matrix)
#                  inPlaneB (the in-plane component of the magnetic field
#                            (Bx for consVarsX, By for consVarsY, Bz for consVarsZ))
#                  gamma (the specific heat ratio of the ideal gas)
#Outputs: primVars (the primitive variables matrix, which also will not include
#                   the in-plane magnetic field component)
def consToPrim_mhd(consVars, inPlaneB, gamma):
    #Create 7xnumYCellsxnumXCells matrix for the primitive variables
    primVars = np.zeros(shape=consVars.shape)
    inverseDensity = 1.0/consVars[0]
    #Density is left unchanged
    primVars[0,:] = consVars[0,:]
    #Convert momenta to velocities
    primVars[1,:] = consVars[1]*inverseDensity
    primVars[2,:] = consVars[2]*inverseDensity
    primVars[3,:] = consVars[3]*inverseDensity
    #Magnetic field components are left unchanged
    primVars[4,:] = consVars[4,:]
    primVars[5,:] = consVars[5,:]
    #Convert total energy to hydrodynamic pressure
    #(P = (gamma-1)*(E - 0.5*rho*V^2 -0.5*B^2) (with permeability of free space u0 set to 1))
    primVars[6] = consVars[6] - 0.5*(consVars[1,:]*consVars[1,:]
                                     + consVars[2,:]*consVars[2,:]
                                     + consVars[3,:]*consVars[3,:])*inverseDensity
    primVars[6] -= 0.5*(inPlaneB*inPlaneB + primVars[4,:]*primVars[4,:]
                                          + primVars[5,:]*primVars[5,:])
    primVars[6,:] *= (gamma-1.0)
    #Return the primitive variables
    return primVars

def consToPrim_mhd_x(consVars, inPlaneB, gamma):
    gammaMinOne = gamma - 1.0
    #Create 7xnumYCellsxnumXCells matrix for the primitive variables
    primVars = np.zeros(shape=consVars.shape)
    inverseDensity = 1.0/consVars[0]
    #Density is left unchanged
    primVars[0,:] = consVars[0,:]
    #Convert momenta to velocities
    primVars[1,:] = consVars[1]*inverseDensity
    primVars[2,:] = consVars[2]*inverseDensity
    primVars[3,:] = consVars[3]*inverseDensity
    #Magnetic field components are left unchanged
    primVars[4,:] = consVars[4,:]
    primVars[5,:] = consVars[5,:]
    #Convert total energy to hydrodynamic pressure
    #(P = (gamma-1)*(E - 0.5*rho*V^2 -0.5*B^2) (with permeability of free space u0 set to 1))
    primVars[6] = consVars[6] - 0.5*(consVars[1,:]*consVars[1,:]
                                     + consVars[2,:]*consVars[2,:]
                                     + consVars[3,:]*consVars[3,:])*inverseDensity
    primVars[6] -= 0.5*(inPlaneB*inPlaneB + primVars[4]*primVars[4] + primVars[5]*primVars[5])
    primVars[6,:] *= gammaMinOne
    #Return the primitive variables
    return primVars

def consToPrim_mhd_y(consVars, inPlaneB, gamma):
    #Create 7xnumYCellsxnumXCells matrix for the primitive variables
    primVars = np.zeros(shape=consVars.shape)
    inverseDensity = 1.0/consVars[0]
    #Density is left unchanged
    primVars[0,:] = consVars[0,:]
    #Convert momenta to velocities
    primVars[1,:] = consVars[1]*inverseDensity
    primVars[2,:] = consVars[2]*inverseDensity
    primVars[3,:] = consVars[3]*inverseDensity
    #Magnetic field components are left unchanged
    primVars[4,:] = consVars[4,:]
    primVars[5,:] = consVars[5,:]
    #Convert total energy to hydrodynamic pressure
    #(P = (gamma-1)*(E - 0.5*rho*V^2 -0.5*B^2) (with permeability of free space u0 set to 1))
    primVars[6] = consVars[6] - 0.5*(consVars[2,:]*consVars[2,:]
                                     + consVars[3,:]*consVars[3,:]
                                     + consVars[1,:]*consVars[1,:])*inverseDensity
    primVars[6] -= 0.5*(inPlaneB*inPlaneB + primVars[4,:]*primVars[4,:]
                                          + primVars[5,:]*primVars[5,:])
    primVars[6,:] *= (gamma-1.0)
    #Return the primitive variables
    return primVars

def consToPrim_mhd_z(consVars, inPlaneB, gamma):
    #Create 7xnumYCellsxnumXCells matrix for the primitive variables
    primVars = np.zeros(shape=consVars.shape)
    inverseDensity = 1.0/consVars[0]
    #Density is left unchanged
    primVars[0,:] = consVars[0,:]
    #Convert momenta to velocities
    primVars[1,:] = consVars[1]*inverseDensity
    primVars[2,:] = consVars[2]*inverseDensity
    primVars[3,:] = consVars[3]*inverseDensity
    #Magnetic field components are left unchanged
    primVars[4,:] = consVars[4,:]
    primVars[5,:] = consVars[5,:]
    #Convert total energy to hydrodynamic pressure
    #(P = (gamma-1)*(E - 0.5*rho*V^2 -0.5*B^2) (with permeability of free space u0 set to 1))
    primVars[6] = consVars[6] - 0.5*(consVars[3,:]*consVars[3,:]
                                     + consVars[1,:]*consVars[1,:]
                                     + consVars[2,:]*consVars[2,:])*inverseDensity
    primVars[6] -= 0.5*(inPlaneB*inPlaneB + primVars[4,:]*primVars[4,:]
                                          + primVars[5,:]*primVars[5,:])
    primVars[6,:] *= (gamma-1.0)
    #Return the primitive variables
    return primVars



#Function: consToPrim_mhd_allVars
#Purpose: Converts conservative variables (density, momenta, magnetic field components, energy) into
#         primitive variables (density, velocities, magnetic field components, pressure) for MHD simulations,
#         and returns the result as a new matrix. This function assumes that the gas is ideal with an
#         adiabatic equation of state.
#         Unlike consToPrim_mhd, consToPrim_mhd_allVars assumes that all magnetic field components
#         are included in the conservative variables matrix (ordered as {density, Mx, My, Mz, Bx, By, Bz,
#         energy}).
#Input Parameters: consVarsAll (the conservative variables matrix,which contains all eight
#                               primitive variables for MHD simulations))
#                  gamma (the specific heat ratio of the ideal gas)
#Outputs: primVars (the primitive variables matrix, which will include all three magnetic field
#                                                   components)
def consToPrim_mhd_allVars(consVarsAll, gamma):
    #Create a matrix for the primitive variables
    primVarsAll = np.zeros(shape=consVarsAll.shape)
    inverseDensity = 1.0/consVarsAll[0]
    #Density is left unchanged
    primVarsAll[0,:] = consVarsAll[0,:]
    #Convert momenta to velocities
    primVarsAll[1,:] = consVarsAll[1,:]*inverseDensity
    primVarsAll[2,:] = consVarsAll[2,:]*inverseDensity
    primVarsAll[3,:] = consVarsAll[3,:]*inverseDensity
    #Magnetic field components are left unchanged
    primVarsAll[4,:] = consVarsAll[4,:]
    primVarsAll[5,:] = consVarsAll[5,:]
    primVarsAll[6,:] = consVarsAll[6,:]
    #Convert total energy to hydrodynamic pressure
    #(P = (gamma-1)*(E - 0.5*rho*V^2 -0.5*B^2) (with permeability of free space u0 set to 1))
    primVarsAll[7] = consVarsAll[7] - 0.5*(consVarsAll[1,:]*consVarsAll[1,:]
                                                        + consVarsAll[2,:]*consVarsAll[2,:]
                                                        + consVarsAll[3,:]*consVarsAll[3,:])*inverseDensity
    primVarsAll[7] -= 0.5*(primVarsAll[4,:]*primVarsAll[4,:]
                                         + primVarsAll[5,:]*primVarsAll[5,:]
                                         + primVarsAll[6,:]*primVarsAll[6,:])
    primVarsAll[7] *= (gamma-1.0)
    #Return the primitive variables
    return primVarsAll

#Function: applyFloors_hydro
#Purpose:
#Input Parameters:
#Outputs:
def applyFloors(vars,isPrimitive,minDens,minPres,minEnergy,calledFromString):
    varString1 = "density"
    minVal1 = minDens
    varString2 = "pressure"
    minVal2 = minPres
    if not isPrimitive:
        varString2 = "energy"
    varString = varString1
    minVal = minVal1
    numDim = len(vars.shape) - 1
    lastVarIndex = vars.shape[0] - 1
    for i in range(2):
        if i == 1:
            varString = varString2
            minVal = minVal2
        if np.any(vars[i*lastVarIndex] < minVal):
            tooLowValIndices = np.where(vars[i*lastVarIndex] < minVal)
            numLowValues = tooLowValIndices[0].shape[0]
            numIndicesToPrint = np.minimum(numLowValues, constants.MAX_INDICES_TO_PRINT)
            errorMessage = "The " + varString + " floor (" + str(minVal) + ") was applied to " + str(numLowValues)\
                            + " in " + calledFromString + ".\nThe " + varString + " values that needed to be corrected"\
                            + " (i.e., increased to the minimum/floor " + varString + " value) were located at the" \
                            + " following indices (up to the first " + str(constants.MAX_INDICES_TO_PRINT)\
                            + " values that were too small):\n"
            if numDim == 1:
                for j in range(numIndicesToPrint):
                    errorMessage += " (x_index = " + str(tooLowValIndices[0][j]) + ", " \
                                    + varString + " = " + str(vars[i*lastVarIndex,tooLowValIndices[0][j]]) + ")"
                    if i < numIndicesToPrint - 1:
                        errorMessage += ","
            elif numDim == 2:
                for j in range(numIndicesToPrint):
                    errorMessage += " (y_index = " + str(tooLowValIndices[0][j]) \
                                       + ", x_index = " + str(tooLowValIndices[1][j]) + ", "\
                                       + varString + " = " + str(vars[i*lastVarIndex,
                                                                  tooLowValIndices[0][j],
                                                                  tooLowValIndices[1][j]]) + ")"
                    if i < numIndicesToPrint - 1:
                        errorMessage += ","
            else:
                for i in range(numIndicesToPrint):
                    errorMessage += " (y_index = " + str(tooLowValIndices[0][i]) \
                                       + ", x_index = " + str(tooLowValIndices[1][i])\
                                       + ", z_index = " + str(tooLowValIndices[2][i]) + ", "\
                                       + varString + " = " + str(vars[i*lastVarIndex,
                                                                  tooLowValIndices[0][i],
                                                                  tooLowValIndices[1][i],
                                                                  tooLowValIndices[2][i]]) + ")"
                    if i < numIndicesToPrint - 1:
                        errorMessage += ","
    vars[0, vars[0] < minDens] = minDens
    if isPrimitive:
        vars[vars.shape[0]-1, vars[vars.shape[0]-1] < minPres] = minPres
    else:
        vars[vars.shape[0]-1, vars[vars.shape[0]-1] < minEnergy] = minEnergy
    return vars
