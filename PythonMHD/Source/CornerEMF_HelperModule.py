#CornerEMF_HelperModule.py
#By Delica Leboe-McGowan, PhD Student, University of Manitoba
#Last Updated: January 1, 2024
#Purpose: Provides functions for calculating the corner EMF values that are required to preserve the divergence-free
#         condition on the magnetic field in an MHD simulation. These functions use the Athena [1] averaging formulas
#         that are described in (Gardiner & Stone, 2005) [2] and (Stone et al., 2008) [3].
#Additional Information: The corner electric field/EMF calculations are part of the Constrained Transport (CT)
#                        component of the CTU+CT integration algorithm that allows us to maintain the divergence-free
#                        condition on magnetic fields while we evolve an MHD gas system with respect to time. PythonMHD's
#                        corner EMF calculations are designed to generate magnetic field updates that are numerically
#                        identical to those produced by Athena [1] (within ), which is why PythonMHD uses the same
#                        averaging formulas that are presented in [2] and [3].
#                        2D Averaging Formula
#
#                        _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
#                        |                            |                            |
#                        |                            |                            |
#                        |                            |                            |
#                        |                            |                            |
#                        |       Ez_(j-1,i-1)   F_(j-1,i-1/2)    Ez_(j-1,i)        |
#                        |                            |                            |
#                        |                            |                            |
#                        |                            |                            |
#                        |                            |                            |
#                        |_ __ _ _G_(j-1/2,i-1)_ _ _ EMF_ _ _  _G_(j-1/2,i)_ _ __ _|
#                        |                            |                            |
#                        |                            |                            |
#                        |                            |                            |
#                        |                            |                            |
#                        |         Ez_(j,i-1)    F_(j,i-1/2)      Ez_(j,i)         |
#                        |                            |                            |
#                        |                            |                            |
#                        |                            |                            |
#                        |                            |                            |
#                        | _ _ _ _ _ _ _ _ _ _ _ _ _ _|_ _ _ _ _ _ _ _ _ _ _ _ _ _ |
#
#                        Ez_(j-1,i-1) = upper-left cell-centred electric field
#                        Ez_(j,i-1) = lower-left cell-centred electric field
#                        Ez_(j-1,i) = upper-right cell-centred electric field
#                        Ez_(j,i) = lower-right cell-centred electric field
#                        F_(j-1,i-1/2) = upper x-direction flux
#                        F_(j,i-1/2) = lower x-direction flux
#                        G_(j-1/2,i-1) = left y-direction flux
#                        G_(j-1/2,i) = right y-direction flux
#
#                        cornerEMF_(j-1/2,i-1/2) = 0.25*(xTerm1 + xTerm2 + yTerm1 + yTerm2)
#
#                        if Vy_(j-1/2,i-1) > 0.0: #if the density flux in G_(j-1/2,i-1) is positive,
#                                                 #which means that the fluid is moving downward at
#                                                 #the (j-1/2,i-1) cell boundary
#                        {
#                           xTerm1 = G_(j-1/2,i-1).BxFlux - F_(j-1,i-1/2).ByFlux - Ez_(j-1,i-1)
#                        }
#                        else if Vy_(j-1/2,i-1) < 0.0: #if the density flux in G_(j-1/2,i-1) is negative,
#                                                      #which means that the fluid is moving upward at
#                                                      #the (j-1/2,i-1) cell boundary
#                        {
#                           xTerm1 = G_(j-1/2,i-1).BxFlux - F_(j,i-1/2).ByFlux - Ez_(j,i-1)
#                        }
#                        else: #if the density flux in G_(j-1/2,i-1) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j-1/2,i-1) cell boundary
#                        {
#                           xTerm1 = G_(j-1/2,i-1).BxFlux - 0.5*(F_(j-1,i-1/2).ByFlux + Ez_(j-1,i-1)
#                                                                + F_(j,i-1/2).ByFlux + Ez_(j,i-1))
#                        }
#
#                        if Vy_(j-1/2,i) > 0.0: #if the density flux in G_(j-1/2,i) is positive,
#                                                 #which means that the fluid is moving downward at
#                                                 #the (j-1/2,i) cell boundary
#                        {
#                           xTerm2 = G_(j-1/2,i).BxFlux - F_(j-1,i-1/2).ByFlux - Ez_(j-1,i)
#                        }
#                        else if Vy_(j-1/2,i) < 0.0: #if the density flux in G_(j-1/2,i) is negative,
#                                                    #which means that the fluid is moving upward at
#                                                    #the (j-1/2,i) cell boundary
#                        {
#                           xTerm2 = G_(j-1/2,i).BxFlux - F_(j,i-1/2).ByFlux - Ez_(j,i)
#                        }
#                        else: #if the density flux in G_(j-1/2,i) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j-1/2,i) cell boundary
#                        {
#                           xTerm2 = G_(j-1/2,i).BxFlux - 0.5*(F_(j-1,i-1/2).ByFlux + Ez_(j-1,i)
#                                                              + F_(j,i-1/2).ByFlux + Ez_(j,i))
#                        }
#
#                        if Vx_(j-1,i-1/2) > 0.0: #if the density flux in F_(j-1,i-1/2) is positive,
#                                                 #which means that the fluid is moving to the right at
#                                                 #the (j-1,i-1/2) cell boundary
#                        {
#                           yTerm1 = -F_(j-1,i-1/2).ByFlux + G_(j-1/2,i-1).BxFlux - Ez_(j-1,i-1)
#                        }
#                        else if Vx_(j-1,i-1/2) < 0.0: #if the density flux in F_(j-1,i-1/2) is negative,
#                                                 #which means that the fluid is moving to the left at
#                                                 #the (j-1,i-1/2) cell boundary
#                        {
#                           yTerm1 = -F_(j-1,i-1/2).ByFlux + G_(j-1/2,i).BxFlux - Ez_(j-1,i)
#                        }
#                        else: #if the density flux in F_(j-1,i-1/2) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j-1,i-1/2) cell boundary
#                        {
#                           yTerm1 = -F_(j-1,i-1/2).ByFlux + 0.5*(G_(j-1/2,i-1).BxFlux - Ez_(j-1,i-1)
#                                                                 + G_(j-1/2,i).BxFlux - Ez_(j-1,i))
#                        }
#
#                        if Vx_(j,i-1/2) > 0.0: #if the density flux in F_(j,i-1/2) is positive,
#                                                 #which means that the fluid is moving to the right at
#                                                 #the (j,i-1/2) cell boundary
#                        {
#                           yTerm2 = -F_(j,i-1/2).ByFlux + G_(j-1/2,i-1).BxFlux - Ez_(j,i-1)
#                        }
#                        else if Vx_(j,i-1/2) < 0.0: #if the density flux in F_(j,i-1/2) is negative,
#                                                 #which means that the fluid is moving to the left at
#                                                 #the (j,i-1/2) cell boundary
#                        {
#                           yTerm2 = -F_(j,i-1/2).ByFlux + G_(j-1/2,i).BxFlux - Ez_(j,i)
#                        }
#                        else: #if the density flux in F_(j,i-1/2) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j,i-1/2) cell boundary
#                        {
#                           yTerm2 = -F_(j,i-1/2).ByFlux + 0.5*(G_(j-1/2,i-1).BxFlux - Ez_(j,i-1)
#                                                                 + G_(j-1/2,i).BxFlux - Ez_(j,i))
#                        }
#
#                        3D Averaging Formula
#                        [Visualizations are not provided for the 3D updates because it is difficult to clearly
#                         present the z-direction in this format. Instead, please refer to PythonMHD's
#                         user guide if you would like to see visualizations of the 3D corner EMF calculations.]
#
#                        CornerEMF_z (z-component of the electric field/EMF at cell corners)
#
#                        Ez_(j-1,i-1,k) = upper-left cell-centred electric field z-component
#                        Ez_(j,i-1,k) = lower-left cell-centred electric field z-component
#                        Ez_(j-1,i,k) = upper-right cell-centred electric field z-component
#                        Ez_(j,i,k) = lower-right cell-centred electric field z-component
#                        F_(j-1,i-1/2,k) = upper x-direction flux
#                        F_(j,i-1/2,k) = lower x-direction flux
#                        G_(j-1/2,i-1,k) = left y-direction flux
#                        G_(j-1/2,i,k) = right y-direction flux
#
#                        cornerEMF_(j-1/2,i-1/2,k-1/2) = 0.25*(xTerm1 + xTerm2 + yTerm1 + yTerm2)
#
#                        if Vy_(j-1/2,i-1,k) > 0.0: #if the density flux in G_(j-1/2,i-1,k) is positive,
#                                                   #which means that the fluid is moving downward at
#                                                   #the (j-1/2,i-1,k) cell boundary
#                        {
#                           xTerm1 = G_(j-1/2,i-1,k).BxFlux - F_(j-1,i-1/2,k).ByFlux - Ez_(j-1,i-1,k)
#                        }
#                        else if Vy_(j-1/2,i-1,k) < 0.0: #if the density flux in G_(j-1/2,i-1,k) is negative,
#                                                        #which means that the fluid is moving upward at
#                                                        #the (j-1/2,i-1,k) cell boundary
#                        {
#                           xTerm1 = G_(j-1/2,i-1,k).BxFlux - F_(j,i-1/2,k).ByFlux - Ez_(j,i-1,k)
#                        }
#                        else: #if the density flux in G_(j-1/2,i-1,k) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j-1/2,i-1,k) cell boundary
#                        {
#                           xTerm1 = G_(j-1/2,i-1,k).BxFlux - 0.5*(F_(j-1,i-1/2,k).ByFlux + Ez_(j-1,i-1,k)
#                                                                + F_(j,i-1/2,k).ByFlux + Ez_(j,i-1,k))
#                        }
#
#                        if Vy_(j-1/2,i,k) > 0.0: #if the density flux in G_(j-1/2,i,k) is positive,
#                                                 #which means that the fluid is moving downward at
#                                                 #the (j-1/2,i,k) cell boundary
#                        {
#                           xTerm2 = G_(j-1/2,i,k).BxFlux - F_(j-1,i-1/2,k).ByFlux - Ez_(j-1,i,k)
#                        }
#                        else if Vy_(j-1/2,i,k) < 0.0: #if the density flux in G_(j-1/2,i,k) is negative,
#                                                    #which means that the fluid is moving upward at
#                                                    #the (j-1/2,i,k) cell boundary
#                        {
#                           xTerm2 = G_(j-1/2,i,k).BxFlux - F_(j,i-1/2,k).ByFlux - Ez_(j,i,k)
#                        }
#                        else: #if the density flux in G_(j-1/2,i,k) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j-1/2,i,k) cell boundary
#                        {
#                           xTerm2 = G_(j-1/2,i,k).BxFlux - 0.5*(F_(j-1,i-1/2).ByFlux + Ez_(j-1,i,k)
#                                                              + F_(j,i-1/2,k).ByFlux + Ez_(j,i,k))
#                        }
#
#                        if Vx_(j-1,i-1/2,k) > 0.0: #if the density flux in F_(j-1,i-1/2,k) is positive,
#                                                 #which means that the fluid is moving to the right at
#                                                 #the (j-1,i-1/2,k) cell boundary
#                        {
#                           yTerm1 = -F_(j-1,i-1/2,k).ByFlux + G_(j-1/2,i-1,k).BxFlux - Ez_(j-1,i-1,k)
#                        }
#                        else if Vx_(j-1,i-1/2,k) < 0.0: #if the density flux in F_(j-1,i-1/2,k) is negative,
#                                                 #which means that the fluid is moving to the left at
#                                                 #the (j-1,i-1/2,k) cell boundary
#                        {
#                           yTerm1 = -F_(j-1,i-1/2,k).ByFlux + G_(j-1/2,i,k).BxFlux - Ez_(j-1,i,k)
#                        }
#                        else: #if the density flux in F_(j-1,i-1/2,k) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j-1,i-1/2,k) cell boundary
#                        {
#                           yTerm1 = -F_(j-1,i-1/2,k).ByFlux + 0.5*(G_(j-1/2,i-1,k).BxFlux - Ez_(j-1,i-1,k)
#                                                                 + G_(j-1/2,i,k).BxFlux - Ez_(j-1,i,k))
#                        }
#
#                        if Vx_(j,i-1/2,k) > 0.0: #if the density flux in F_(j,i-1/2,k) is positive,
#                                                 #which means that the fluid is moving to the right at
#                                                 #the (j,i-1/2,k) cell boundary
#                        {
#                           yTerm2 = -F_(j,i-1/2,k).ByFlux + G_(j-1/2,i-1,k).BxFlux - Ez_(j,i-1,k)
#                        }
#                        else if Vx_(j,i-1/2,k) < 0.0: #if the density flux in F_(j,i-1/2,k) is negative,
#                                                      #which means that the fluid is moving to the left at
#                                                      #the (j,i-1/2,k) cell boundary
#                        {
#                           yTerm2 = -F_(j,i-1/2,k).ByFlux + G_(j-1/2,i,k).BxFlux - Ez_(j,i,k)
#                        }
#                        else: #if the density flux in F_(j,i-1/2,k) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j,i-1/2,k) cell boundary
#                        {
#                           yTerm2 = -F_(j,i-1/2,k).ByFlux + 0.5*(G_(j-1/2,i-1,k).BxFlux - Ez_(j,i-1,k)
#                                                                 + G_(j-1/2,i,k).BxFlux - Ez_(j,i,k))
#                        }
#
#                        CornerEMF_x (x-component of the electric field/EMF at cell corners)
#
#                        Ex_(j-1,i,k-1) = upper-back cell-centred electric field x-component
#                        Ex_(j,i,k-1) = lower-back cell-centred electric field x-component
#                        Ex_(j-1,i,k) = upper-forward cell-centred electric field x-component
#                        Ex_(j,i,k) = lower-forward cell-centred electric field x-component
#                        H_(j-1,i,k-1/2) = upper z-direction flux
#                        H_(j,i,k-1/2) = lower z-direction flux
#                        G_(j-1/2,i,k-1) = back y-direction flux
#                        G_(j-1/2,i,k) = forward y-direction flux
#
#                        cornerEMF_(j-1/2,i-1/2,k-1/2) = 0.25*(zTerm1 + zTerm2 + yTerm1 + yTerm2)
#
#                        if Vy_(j-1/2,i,k-1) > 0.0: #if the density flux in G_(j-1/2,i,k-1) is positive,
#                                                   #which means that the fluid is moving downward at
#                                                   #the (j-1/2,i,k-1) cell boundary
#                        {
#                           zTerm1 = -G_(j-1/2,i,k-1).BzFlux + H_(j-1,i,k-1/2).ByFlux - Ex_(j-1,i,k-1)
#                        }
#                        else if Vy_(j-1/2,i,k-1) < 0.0: #if the density flux in G_(j-1/2,i,k-1) is negative,
#                                                        #which means that the fluid is moving downward at
#                                                        #the (j-1/2,i,k-1) cell boundary
#                        {
#                           zTerm1 = -G_(j-1/2,i,k-1).BzFlux + H_(j,i,k-1/2).ByFlux - Ex_(j,i,k-1)
#                        }
#                        else: #if the density flux in G_(j-1/2,i,k-1) is zero,
#                              #which means that the fluid is stationary at
#                             #the (j-1/2,i,k-1) cell boundary
#                        {
#                           zTerm1 = -G_(j-1/2,i,k-1).BzFlux + (H_(j-1,i,k-1/2).ByFlux - Ex_(j-1,i,k-1)
#                                                               + H_(j,i,k-1/2).ByFlux - Ex_(j,i,k-1))
#                        }
#
#                        if Vy_(j-1/2,i,k) > 0.0: #if the density flux in G_(j-1/2,i,k) is positive,
#                                                 #which means that the fluid is moving downward at
#                                                 #the (j-1/2,i,k) cell boundary
#                        {
#                           zTerm2 = -G_(j-1/2,i,k).BzFlux + H_(j-1,i,k-1/2).ByFlux - Ex_(j-1,i,k-1)
#                        }
#                        else if Vy_(j-1/2,i,k) < 0.0: #if the density flux in G_(j-1/2,i,k-1) is negative,
#                                                        #which means that the fluid is moving downward at
#                                                        #the (j-1/2,i,k-1) cell boundary
#                        {
#                           zTerm2 = -G_(j-1/2,i,k).BzFlux + H_(j,i,k-1/2).ByFlux - Ex_(j,i,k-1)
#                        }
#                        else: #if the density flux in G_(j-1/2,i,k-1) is zero,
#                              #which means that the fluid is stationary at
#                             #the (j-1/2,i,k-1) cell boundary
#                        {
#                           zTerm2 = -G_(j-1/2,i,k).BzFlux + 0.5*(H_(j-1,i,k-1/2).ByFlux - Ex_(j-1,i,k)
#                                                               + H_(j,i,k-1/2).ByFlux - Ex_(j,i,k))
#                        }
#
#                        if Vz_(j-1,i,k-1/2) > 0.0: #if the density flux in H_(j-1,i,k-1/2) is positive,
#                                                   #which means that the fluid is moving into the screen at
#                                                   #the (j-1,i,k-1/2) cell boundary
#                        {
#                           yTerm1 = H_(j-1,i,k-1/2).ByFlux - G_(j-1/2,i,k-1).BzFlux - Ex_(j-1,i,k-1)
#                        }
#                        else if Vz_(j-1,i,k-1/2) < 0.0: #if the density flux in H_(j-1,i,k-1/2) is negative,
#                                                        #which means that the fluid is moving out of the screen at
#                                                        #the (j-1,i,k-1/2) cell boundary
#                        {
#                           yTerm1 = H_(j-1,i,k-1/2).ByFlux - G_(j-1/2,i,k).BzFlux - Ex_(j-1,i,k)
#                        }
#                        else: #if the density flux in H_(j-1,i,k-1/2) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j-1,i,k-1/2) cell boundary
#                        {
#                           yTerm1 = H_(j-1,i,k-1/2).ByFlux - 0.5*(G_(j-1/2,i,k-1).BzFlux + Ex_(j-1,i,k-1)
#                                                                  + G_(j-1/2,i,k).BzFlux + Ex_(j-1,i,k))
#                        }
#
#                        if Vz_(j,i,k-1/2) > 0.0: #if the density flux in H_(j,i,k-1/2) is positive,
#                                                   #which means that the fluid is moving into the screen at
#                                                   #the (j,i,k-1/2) cell boundary
#                        {
#                           yTerm2 = H_(j,i,k-1/2).ByFlux - G_(j-1/2,i,k-1).BzFlux - Ex_(j,i,k-1)
#                        }
#                        else if Vz_(j,i,k-1/2) < 0.0: #if the density flux in H_(j,i,k-1/2) is negative,
#                                                      #which means that the fluid is moving out of the screen at
#                                                      #the (j,i,k-1/2) cell boundary
#                        {
#                           yTerm2 = H_(j,i,k-1/2).ByFlux - G_(j-1/2,i,k).BzFlux - Ex_(j,i,k)
#                        }
#                        else: #if the density flux in H_(j,i,k-1/2) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j,i,k-1/2) cell boundary
#                        {
#                           yTerm2 = H_(j,i,k-1/2).ByFlux - 0.5*(G_(j-1/2,i,k-1).BzFlux + Ex_(j,i,k-1)
#                                                                  + G_(j-1/2,i,k).BzFlux + Ex_(j,i,k))
#                        }
#
#                        CornerEMF_y (y-component of the electric field/EMF at cell corners)
#
#                        Ey_(j,i-1,k-1) = left-back cell-centred electric field y-component
#                        Ey_(j,i,k-1) = right-back cell-centred electric field y-component
#                        Ey_(j,i-1,k) = left-forward cell-centred electric field y-component
#                        Ey_(j,i,k) = right-forward cell-centred electric field x-component
#                        H_(j,i-1,k-1/2) = left z-direction flux
#                        H_(j,i,k-1/2) = right z-direction flux
#                        F_(j,i-1/2,k-1) = back x-direction flux
#                        F_(j,i-1/2,k) = forward x-direction flux
#
#                        cornerEMF_(j-1/2,i-1/2,k-1/2) = 0.25*(zTerm1 + zTerm2 + xTerm1 + xTerm2)
#
#                        if Vx_(j,i-1/2,k-1) > 0.0: #if the density flux in F_(j,i-1/2,k-1) is positive,
#                                                   #which means that the fluid is moving to the right at
#                                                   #the (j,i-1/2,k-1) cell boundary
#                        {
#                           zTerm1 = F_(j,i-1/2,k-1).BzFlux - H_(j,i-1,k-1/2).BxFlux - Ex_(j,i-1,k-1)
#                        }
#                        else if Vx_(j,i-1/2,k-1) < 0.0: #if the density flux in F_(j,i-1/2,k-1) is negative,
#                                                        #which means that the fluid is moving to the left at
#                                                        #the (j,i-1/2,k-1) cell boundary
#                        {
#                           zTerm1 = F_(j,i-1/2,k-1).BzFlux - H_(j,i,k-1/2).BxFlux - Ex_(j,i,k-1)
#                        }
#                        else: #if the density flux in F_(j,i-1/2,k-1) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j,i-1/2,k-1) cell boundary
#                        {
#                           zTerm1 = F_(j,i-1/2,k-1).BzFlux - 0.5*(H_(j,i-1,k-1/2).BxFlux + Ex_(j,i-1,k-1)
#                                                                   + H_(j,i,k-1/2).BxFlux + Ex_(j,i,k-1))
#                        }
#
#                        if Vx_(j,i-1/2,k) > 0.0: #if the density flux in F_(j,i-1/2,k) is positive,
#                                                 #which means that the fluid is moving to the right at
#                                                 #the (j,i-1/2,k) cell boundary
#                        {
#                           zTerm2 = F_(j,i-1/2,k).BzFlux - H_(j,i-1,k-1/2).BxFlux - Ex_(j,i-1,k)
#                        }
#                        else if Vx_(j,i-1/2,k) < 0.0: #if the density flux in F_(j,i-1/2,k) is negative,
#                                                      #which means that the fluid is moving to the left at
#                                                      #the (j,i-1/2,k) cell boundary
#                        {
#                           zTerm2 = F_(j,i-1/2,k).BzFlux - H_(j,i,k-1/2).BxFlux - Ex_(j,i,k)
#                        }
#                        else: #if the density flux in F_(j,i-1/2,k) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j,i-1/2,k) cell boundary
#                        {
#                           zTerm2 = F_(j,i-1/2,k).BzFlux - 0.5*(H_(j,i-1,k-1/2).BxFlux + Ex_(j,i-1,k)
#                                                                   + H_(j,i,k-1/2).BxFlux + Ex_(j,i,k))
#                        }
#
#                        if Vz_(j,i-1,k-1/2) > 0.0: #if the density flux in H_(j,i-1,k-1/2) is positive,
#                                                   #which means that the fluid is moving into the screen at
#                                                   #the (j,i-1,k-1/2) cell boundary
#                        {
#                           xTerm1 = -H_(j,i-1,k-1/2).BxFlux + F_(j,i-1/2,k-1).BzFlux - Ex_(j,i-1,k-1)
#                        }
#                        else if Vz_(j,i-1,k-1/2) < 0.0: #if the density flux in H_(j,i-1,k-1/2) is negative,
#                                                        #which means that the fluid is moving out of the screen at
#                                                        #the (j,i-1,k-1/2) cell boundary
#                        {
#                           xTerm1 = -H_(j,i-1,k-1/2).BxFlux + F_(j,i-1/2,k).BzFlux - Ex_(j,i-1,k)
#                        }
#                        else: #if the density flux in H_(j,i-1,k-1/2) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j,i-1,k-1/2) cell boundary
#                        {
#                           xTerm1 = -H_(j,i-1,k-1/2).BxFlux + 0.5*(F_(j,i-1/2,k).BzFlux - Ex_(j,i-1,k)
#                                                                    + F_(j,i-1/2,k).BzFlux - Ex_(j,i-1,k))
#                        }
#
#                        if Vz_(j,i,k-1/2) > 0.0: #if the density flux in H_(j,i,k-1/2) is positive,
#                                                 #which means that the fluid is moving into the screen at
#                                                 #the (j,i,k-1/2) cell boundary
#                        {
#                           xTerm2 = -H_(j,i,k-1/2).BxFlux + F_(j,i-1/2,k-1).BzFlux - Ex_(j,i,k-1)
#                        }
#                        else if Vz_(j,i,k-1/2) < 0.0: #if the density flux in H_(j,i,k-1/2) is negative,
#                                                      #which means that the fluid is moving out of the screen at
#                                                      #the (j,i,k-1/2) cell boundary
#                        {
#                           xTerm2 = -H_(j,i,k-1/2).BxFlux + F_(j,i-1/2,k).BzFlux - Ex_(j,i,k)
#                        }
#                        else: #if the density flux in H_(j,i,k-1/2) is zero,
#                              #which means that the fluid is stationary at
#                              #the (j,i,k-1/2) cell boundary
#                        {
#                           xTerm2 = -H_(j,i,k-1/2).BxFlux + 0.5*(F_(j,i-1/2,k).BzFlux - Ex_(j,i,k)
#                                                                 + F_(j,i-1/2,k).BzFlux - Ex_(j,i,k))
#                        }
#
#
#References:
# 1. https://github.com/PrincetonUniversity/Athena-Cversion
# 2. Stone, J. M., Gardiner, T. A., Teuben, P., Hawley, J. F., & Simon, J. B. (2008).
#    Athena: A new code for astrophysical MHD. The Astrophysical Journal Supplemental Series,
#    178(1), 137-177. https://iopscience.iop.org/article/10.1086/588755/pdf
# 3. Gardiner, T. A., & Stone, J. M. (2005). An unsplit Godunov method for ideal MHD via Constrained Transport.
#    Journal of Computational Physics, 205(2), 509-539. https://arxiv.org/pdf/astro-ph/0501557.pdf

######IMPORT STATEMENTS######

#Import NumPy for matrix operations
import numpy as np

#####2D CORNER EMF FUNCTION#####

#Function: getCornerEMF_2D
#Purpose: Calculates all of the corner EMF values that are required for
#         updating face-centred magnetic fields in a 2D MHD simulation.
#Input Parameters: interCellFluxX (the intercell flux vectors in the x-direction)
#                  interCellFluxY (the intercell flux vectors in the y-direction)
#                  Ez (the z-component of the electric field at the centre
#                      of each cell)
#                  BcX (integer for the boundary condition in the x-direction
#                      (0 = outflow and 1 = periodic))
#                  BcY (integer for the boundary condition in the x-direction
#                      (0 = outflow and 1 = periodic))
#Outputs: cornerEMF (a matrix with the EMFs for every cell corner in the simulation grid)
def getCornerEMF_2D(interCellFluxX, interCellFluxY, Ez, BcX, BcY):
    topFlux = np.copy(interCellFluxY)
    if BcX == 0:
        topFlux = np.append(topFlux,topFlux[:,:,topFlux.shape[2]-1].reshape(7,-1,1),axis=2)
    else:
        topFlux = np.append(topFlux,topFlux[:,:,0].reshape(7,-1,1),axis=2)
    if BcX == 0:
        leftTopFlux = np.append(interCellFluxY[:,:,0].reshape(7,-1,1),interCellFluxY,axis=2)
    else:
        leftTopFlux = np.append(interCellFluxY[:,:,interCellFluxY.shape[2]-1].reshape(7,-1,1),interCellFluxY,axis=2)
    leftFlux = np.copy(interCellFluxX)
    if BcY == 0:
        leftFlux = np.append(leftFlux,leftFlux[:,leftFlux.shape[1]-1,:].reshape(7,1,-1),axis=1)
    else:
        leftFlux = np.append(leftFlux,leftFlux[:,0].reshape(7,1,-1),axis=1)
    if BcY == 0:
        upperLeftFlux = np.append(interCellFluxX[:,0,:].reshape(7,1,-1),interCellFluxX,axis=1)
    else:
        upperLeftFlux = np.append(interCellFluxX[:,interCellFluxX.shape[1]-1,:].reshape(7,1,-1),
                                  interCellFluxX,axis=1)

    centEField = np.copy(Ez)
    leftCentEField = np.copy(Ez)
    upperCentEField = np.copy(Ez)
    upperLeftCentEField = np.copy(Ez)
    if BcX == 0:
        centEField = np.append(centEField,centEField[:,centEField.shape[1]-1].reshape(-1,1),axis=1)
        leftCentEField = np.append(leftCentEField[:,0].reshape(-1,1),leftCentEField,axis=1)
        upperCentEField = np.append(upperCentEField,
                                    upperCentEField[:,upperCentEField.shape[1]-1].reshape(-1,1),axis=1)
        upperLeftCentEField = np.append(upperLeftCentEField[:,0].reshape(-1,1),upperLeftCentEField,axis=1)
    else:
        centEField = np.append(centEField,centEField[:,0].reshape(-1,1),axis=1)
        leftCentEField = np.append(leftCentEField[:,leftCentEField.shape[1]-1].reshape(-1,1),
                                   leftCentEField,axis=1)
        upperCentEField = np.append(upperCentEField,
                                    upperCentEField[:,0].reshape(-1,1),axis=1)
        upperLeftCentEField = np.append(upperLeftCentEField[:,upperLeftCentEField.shape[1]-1].reshape(-1,1),
                                        upperLeftCentEField,axis=1)
    if BcY == 0:
        centEField = np.append(centEField,centEField[centEField.shape[0]-1,:].reshape(1,-1),axis=0)
        upperCentEField = np.append(upperCentEField[0,:].reshape(1,-1),upperCentEField,axis=0)
        upperLeftCentEField = np.append(upperLeftCentEField[0,:].reshape(1,-1),upperLeftCentEField,axis=0)
        leftCentEField = np.append(leftCentEField,
                                   leftCentEField[leftCentEField.shape[0]-1,:].reshape(1,-1),axis=0)
    else:
        centEField = np.append(centEField,centEField[0,:].reshape(1,-1),axis=0)
        upperCentEField = np.append(upperCentEField[upperCentEField.shape[0]-1,:].reshape(1,-1),
                                    upperCentEField,axis=0)
        upperLeftCentEField = np.append(upperLeftCentEField[upperLeftCentEField.shape[0]-1,:].reshape(1,-1),
                                        upperLeftCentEField,axis=0)
        leftCentEField = np.append(leftCentEField,
                                   leftCentEField[0,:].reshape(1,-1),axis=0)

    #Calculate a term that is proportional to the change in Ez
    #from the left interface (i - 1/2) to the centre of cell i - 1
    xTerm1 = np.zeros(shape=(topFlux.shape[1], topFlux.shape[2]))
    xTerm1[leftTopFlux[0] > 0.0] = (leftTopFlux[5] - upperLeftFlux[4] - upperLeftCentEField)[leftTopFlux[0] > 0.0]
    xTerm1[leftTopFlux[0] < 0.0] = (leftTopFlux[5] - leftFlux[4] - leftCentEField)[leftTopFlux[0] < 0.0]
    xTerm1[leftTopFlux[0] == 0.0] = (leftTopFlux[5]
                                     - 0.5 * (leftFlux[4] + upperLeftFlux[4]
                                              + leftCentEField + upperLeftCentEField))[leftTopFlux[0] == 0.0]
    #Calculate a term that is proportional to the change in Ez
    #from the left interface (i - 1/2) to the centre of cell i
    xTerm2 = np.zeros(shape=(topFlux.shape[1], topFlux.shape[2]))
    xTerm2[topFlux[0] > 0.0] = (topFlux[5] - upperLeftFlux[4] - upperCentEField)[topFlux[0] > 0.0]
    xTerm2[topFlux[0] < 0.0] = (topFlux[5] - leftFlux[4] - centEField)[topFlux[0] < 0.0]
    xTerm2[topFlux[0] == 0.0] = (topFlux[5] - 0.5 * (leftFlux[4] + upperLeftFlux[4]
                                                     + centEField + upperCentEField))[topFlux[0] == 0.0]

    #Calculate a term that is proportional to the change in Ez
    #from the top interface (j - 1/2) to the centre of cell j - 1
    yTerm1 = np.zeros(shape=(topFlux.shape[1], topFlux.shape[2]))
    yTerm1[upperLeftFlux[0] > 0.0] = (-upperLeftFlux[4] + leftTopFlux[5] - upperLeftCentEField)[upperLeftFlux[0] > 0.0]
    yTerm1[upperLeftFlux[0] < 0.0] = (-upperLeftFlux[4] + topFlux[5] - upperCentEField)[upperLeftFlux[0] < 0.0]
    yTerm1[upperLeftFlux[0] == 0.0] = (-upperLeftFlux[4]
                                       + 0.5 * (leftTopFlux[5] + topFlux[5]
                                                - upperLeftCentEField - upperCentEField))[upperLeftFlux[0] == 0.0]

    #Calculate a term that is proportional to the change in Ez
    #from the top interface (j - 1/2) to the centre of cell j
    yTerm2 = np.zeros(shape=(topFlux.shape[1], topFlux.shape[2]))
    yTerm2[leftFlux[0] > 0.0] = (-leftFlux[4] + leftTopFlux[5] - leftCentEField)[leftFlux[0] > 0.0]
    yTerm2[leftFlux[0] < 0.0] = (-leftFlux[4] + topFlux[5] - centEField)[leftFlux[0] < 0.0]
    yTerm2[leftFlux[0] == 0.0] = (-leftFlux[4] + 0.5 * (leftTopFlux[5] + topFlux[5]
                                                        - leftCentEField - centEField))[leftFlux[0] == 0.0]

    # Take the average of the four terms to get the Ez value at the upper left corner of each cell
    cornerEMF = 0.25 * (xTerm1 + xTerm2 + yTerm1 + yTerm2)
    return cornerEMF

#####3D CORNER EMF FUNCTION (Z-COMPONENT)######
#Function: getZCornerEMF_3D
#Purpose: Calculates the z-component of the electric field/EMF at every cell corner
#         in a 3D simulation.
#Input Parameters: interCellFluxX (the intercell flux vectors in the x-direction)
#                  interCellFluxY (the intercell flux vectors in the y-direction)
#                  Ez (the z-component of the electric field at the centre
#                      of each cell)
#                  BcX (integer for the boundary condition in the x-direction
#                      (0 = outflow and 1 = periodic))
#                  BcY (integer for the boundary condition in the x-direction
#                      (0 = outflow and 1 = periodic))
#                  BcZ (integer for the boundary condition in the z-direction
#                      (0 = outflow and 1 = periodic))
#Outputs: cornerEMF (a matrix with the z-component of the EMF at every cell corner in the simulation grid)
def getZCornerEMF_3D(interCellFluxX, interCellFluxY, Ez, BcX, BcY, BcZ):
    topFlux = np.copy(interCellFluxY)
    if BcX == 0:
        topFlux = np.append(topFlux,topFlux[:,:,topFlux.shape[2]-1,:].reshape(7,topFlux.shape[1],1,-1),axis=2)
    else:
        topFlux = np.append(topFlux,topFlux[:,:,0,:].reshape(7,topFlux.shape[1],1,-1),axis=2)
    if BcX == 0:
        leftTopFlux = np.append(interCellFluxY[:,:,0,:].reshape(7,interCellFluxY.shape[1],1,-1),interCellFluxY,axis=2)
    else:
        leftTopFlux = np.append(interCellFluxY[:,:,interCellFluxY.shape[2]-1,:].reshape(7,interCellFluxY.shape[1],1,-1),interCellFluxY,axis=2)
    leftFlux = np.copy(interCellFluxX)
    if BcY == 0:
        leftFlux = np.append(leftFlux,leftFlux[:,leftFlux.shape[1]-1,:,:].reshape(7,1,leftFlux.shape[2],-1),axis=1)
    else:
        leftFlux = np.append(leftFlux,leftFlux[:,0,:,:].reshape(7,1,leftFlux.shape[2],-1),axis=1)
    if BcY == 0:
        upperLeftFlux = np.append(interCellFluxX[:,0,:,:].reshape(7,1,interCellFluxX.shape[2],-1),interCellFluxX,axis=1)
    else:
        upperLeftFlux = np.append(interCellFluxX[:,interCellFluxX.shape[1]-1,:,:].reshape(7,1,interCellFluxX.shape[2],-1),
                                  interCellFluxX,axis=1)
    if BcZ == 0:
        leftFlux = np.append(leftFlux,leftFlux[:,:,:,leftFlux.shape[3]-1].reshape(7,-1,leftFlux.shape[2],1),axis=3)
        upperLeftFlux = np.append(upperLeftFlux,
                                  upperLeftFlux[:,:,:,upperLeftFlux.shape[3]-1].reshape(7,-1,upperLeftFlux.shape[2],1),
                                  axis=3)
        topFlux = np.append(topFlux,topFlux[:,:,:,topFlux.shape[3]-1].reshape(7,-1,topFlux.shape[2],1),axis=3)
        leftTopFlux = np.append(leftTopFlux,leftTopFlux[:,:,:,leftTopFlux.shape[3]-1].reshape(7,-1,leftTopFlux.shape[2],1),axis=3)
    else:
        leftFlux = np.append(leftFlux,leftFlux[:,:,:,0].reshape(7,-1,leftFlux.shape[2],1),axis=3)
        upperLeftFlux = np.append(upperLeftFlux,upperLeftFlux[:,:,:,0].reshape(7,-1,upperLeftFlux.shape[2],1),axis=3)
        topFlux = np.append(topFlux,topFlux[:,:,:,0].reshape(7,-1,topFlux.shape[2],1),axis=3)
        leftTopFlux = np.append(leftTopFlux,leftTopFlux[:,:,:,0].reshape(7,-1,leftTopFlux.shape[2],1),axis=3)
    centEField = np.copy(Ez)
    leftCentEField = np.copy(Ez)
    upperCentEField = np.copy(Ez)
    upperLeftCentEField = np.copy(Ez)
    if BcX == 0:
        centEField = np.append(centEField,centEField[:,centEField.shape[1]-1,:].reshape(centEField.shape[0],1,-1),axis=1)
        leftCentEField = np.append(leftCentEField[:,0,:].reshape(leftCentEField.shape[0],1,-1),leftCentEField,axis=1)
        upperCentEField = np.append(upperCentEField,
                                    upperCentEField[:,upperCentEField.shape[1]-1,:].reshape(upperCentEField.shape[0],1,-1),
                                    axis=1)
        upperLeftCentEField = np.append(upperLeftCentEField[:,0,:].reshape(upperLeftCentEField.shape[0],1,-1),
                                        upperLeftCentEField,axis=1)
    else:
        centEField = np.append(centEField,centEField[:,0,:].reshape(centEField.shape[0],1,-1),axis=1)
        leftCentEField = np.append(leftCentEField[:,leftCentEField.shape[1]-1,:].reshape(leftCentEField.shape[0],1,-1),
                                   leftCentEField,axis=1)
        upperCentEField = np.append(upperCentEField,
                                    upperCentEField[:,0,:].reshape(upperCentEField.shape[0],1,-1),axis=1)
        upperLeftCentEField = np.append(upperLeftCentEField[:,upperLeftCentEField.shape[1]-1,:].reshape(upperLeftCentEField.shape[0],1,-1),
                                        upperLeftCentEField,axis=1)
    if BcY == 0:
        centEField = np.append(centEField,centEField[centEField.shape[0]-1,:,:].reshape(1,centEField.shape[1],-1),axis=0)
        upperCentEField = np.append(upperCentEField[0,:,:].reshape(1,upperCentEField.shape[1],-1),upperCentEField,axis=0)
        upperLeftCentEField = np.append(upperLeftCentEField[0,:,:].reshape(1,upperLeftCentEField.shape[1],-1),
                                        upperLeftCentEField,axis=0)
        leftCentEField = np.append(leftCentEField,
                                   leftCentEField[leftCentEField.shape[0]-1,:,:].reshape(1,leftCentEField.shape[1],-1),axis=0)
    else:
        centEField = np.append(centEField,centEField[0,:,:].reshape(1,centEField.shape[1],-1),axis=0)
        upperCentEField = np.append(upperCentEField[upperCentEField.shape[0]-1,:,:].reshape(1,upperCentEField.shape[1],-1),
                                    upperCentEField,axis=0)
        upperLeftCentEField = np.append(upperLeftCentEField[upperLeftCentEField.shape[0]-1,:,:].reshape(1,upperLeftCentEField.shape[1],-1),
                                        upperLeftCentEField,axis=0)
        leftCentEField = np.append(leftCentEField,
                                   leftCentEField[0,:,:].reshape(1,leftCentEField.shape[1],-1),axis=0)
    if BcZ == 0:
        centEField = np.append(centEField,centEField[:,:,centEField.shape[2]-1].reshape(-1,centEField.shape[1],1),axis=2)
        upperCentEField = np.append(upperCentEField,
                                    upperCentEField[:,:,upperCentEField.shape[2]-1].reshape(-1,upperCentEField.shape[1],1),axis=2)
        upperLeftCentEField = np.append(upperLeftCentEField,
                                        upperLeftCentEField[:,:,upperLeftCentEField.shape[2]-1].reshape(-1,upperLeftCentEField.shape[1],1),axis=2)
        leftCentEField = np.append(leftCentEField,
                                   leftCentEField[:,:,leftCentEField.shape[2]-1].reshape(-1,leftCentEField.shape[1],1),
                                   axis=2)
    else:
        centEField = np.append(centEField,centEField[:,:,0].reshape(-1,centEField.shape[1],1),axis=2)
        upperCentEField = np.append(upperCentEField,
                                    upperCentEField[:,:,0].reshape(-1,upperCentEField.shape[1],1),
                                    axis=2)
        upperLeftCentEField = np.append(upperLeftCentEField,
                                        upperLeftCentEField[:,:,0].reshape(-1,upperLeftCentEField.shape[1],1),
                                        axis=2)
        leftCentEField = np.append(leftCentEField,
                                   leftCentEField[:,:,0].reshape(-1,leftCentEField.shape[1],1),axis=2)

    #Calculate a term that is proportional to the change in Ez
    #from the left interface (i - 1/2) to the centre of cell i - 1
    xTerm1 = np.zeros(shape=(topFlux.shape[1], topFlux.shape[2], topFlux.shape[3]))
    xTerm1[leftTopFlux[0] > 0.0] = (-upperLeftFlux[4] - upperLeftCentEField)[leftTopFlux[0] > 0.0]
    xTerm1[leftTopFlux[0] < 0.0] = (-leftFlux[4] - leftCentEField)[leftTopFlux[0] < 0.0]
    xTerm1[leftTopFlux[0] == 0.0] = (-0.5 * (leftFlux[4] + upperLeftFlux[4]
                                              + leftCentEField + upperLeftCentEField))[leftTopFlux[0] == 0.0]
    #Calculate a term that is proportional to the change in Ez
    #from the left interface (i - 1/2) to the centre of cell i
    xTerm2 = np.zeros(shape=(topFlux.shape[1], topFlux.shape[2], topFlux.shape[3]))
    xTerm2[topFlux[0] > 0.0] = (-upperLeftFlux[4] - upperCentEField)[topFlux[0] > 0.0]
    xTerm2[topFlux[0] < 0.0] = (-leftFlux[4] - centEField)[topFlux[0] < 0.0]
    xTerm2[topFlux[0] == 0.0] = (-0.5 * (leftFlux[4] + upperLeftFlux[4]
                                                     + centEField + upperCentEField))[topFlux[0] == 0.0]

    #Calculate a term that is proportional to the change in Ez
    #from the top interface (j - 1/2) to the centre of cell j - 1
    yTerm1 = np.zeros(shape=(topFlux.shape[1], topFlux.shape[2], topFlux.shape[3]))
    yTerm1[upperLeftFlux[0] > 0.0] = (leftTopFlux[5] - upperLeftCentEField)[
        upperLeftFlux[0] > 0.0]
    yTerm1[upperLeftFlux[0] < 0.0] = (topFlux[5] - upperCentEField)[upperLeftFlux[0] < 0.0]
    yTerm1[upperLeftFlux[0] == 0.0] = (0.5 * (leftTopFlux[5] + topFlux[5]
                                                - upperLeftCentEField - upperCentEField))[upperLeftFlux[0] == 0.0]

    #Calculate a term that is proportional to the change in Ez
    #from the top interface (j - 1/2) to the centre of cell j
    yTerm2 = np.zeros(shape=(topFlux.shape[1], topFlux.shape[2], topFlux.shape[3]))
    yTerm2[leftFlux[0] > 0.0] = (leftTopFlux[5] - leftCentEField)[leftFlux[0] > 0.0]
    yTerm2[leftFlux[0] < 0.0] = (topFlux[5] - centEField)[leftFlux[0] < 0.0]
    yTerm2[leftFlux[0] == 0.0] = (0.5 * (leftTopFlux[5] + topFlux[5]
                                                        - leftCentEField - centEField))[leftFlux[0] == 0.0]

    #Take the average of the four terms to get the Ez value at each cell corner
    cornerEMF = 0.25 * (leftTopFlux[5] + topFlux[5] -upperLeftFlux[4] - leftFlux[4]
                        + xTerm1 + xTerm2 + yTerm1 + yTerm2)
    return cornerEMF

######3D CORNER EMF FUNCTION (X-COMPONENT)######

#Function: getXCornerEMF_3D
#Purpose: Calculates the x-component of the electric field/EMF at every cell corner
#         in a 3D simulation.
#Input Parameters: interCellFluxY (the intercell flux vectors in the y-direction)
#                  interCellFluxZ (the intercell flux vectors in the z-direction)
#                  Ex (the x-component of the electric field at the centre
#                      of each cell)
#                  BcX (integer for the boundary condition in the x-direction
#                      (0 = outflow and 1 = periodic))
#                  BcY (integer for the boundary condition in the x-direction
#                      (0 = outflow and 1 = periodic))
#                  BcZ (integer for the boundary condition in the z-direction
#                      (0 = outflow and 1 = periodic))
#Outputs: cornerEMF (a matrix with the x-component of the EMF at every cell corner in the simulation grid)
def getXCornerEMF_3D(interCellFluxY, interCellFluxZ, Ex, BcX, BcY, BcZ):
    backFlux = np.copy(interCellFluxZ)
    if BcX == 0:
        backFlux = np.append(backFlux,backFlux[:,:,backFlux.shape[2]-1,:].reshape(7,backFlux.shape[1],1,-1),axis=2)
    else:
        backFlux = np.append(backFlux,backFlux[:,:,0,:].reshape(7,backFlux.shape[1],1,-1),axis=2)
    if BcY == 0:
        upperBackFlux = np.append(backFlux[:,0,:,:].reshape(7,1,backFlux.shape[2],-1),backFlux,axis=1)
        backFlux = np.append(backFlux,
                             backFlux[:,backFlux.shape[1]-1,:,:].reshape(7,1,backFlux.shape[2],-1),
                             axis=1)
    else:
        upperBackFlux = np.append(backFlux[:,backFlux.shape[1]-1,:,:].reshape(7,1,backFlux.shape[2],-1),backFlux,axis=1)
        backFlux = np.append(backFlux,
                             backFlux[:,0,:,:].reshape(7,1,-1,backFlux.shape[3]),
                             axis=1)
    topFlux = np.copy(interCellFluxY)
    if BcX == 0:
        topFlux = np.append(topFlux,topFlux[:,:,topFlux.shape[2]-1,:].reshape(7,topFlux.shape[1],1,-1),axis=2)
    else:
        topFlux = np.append(topFlux,topFlux[:,:,0,:].reshape(7,topFlux.shape[1],1,-1),axis=2)
    if BcZ == 0:
        backTopFlux = np.append(topFlux[:,:,:,0].reshape(7,topFlux.shape[1],-1,1),topFlux,axis=3)
        topFlux = np.append(topFlux,topFlux[:,:,:,topFlux.shape[3]-1].reshape(7,topFlux.shape[1],-1,1),axis=3)
    else:
        backTopFlux = np.append(topFlux[:,:,:,topFlux.shape[3]-1].reshape(7,topFlux.shape[1],-1,1),topFlux,axis=3)
        topFlux = np.append(topFlux,topFlux[:,:,:,0].reshape(7,topFlux.shape[1],-1,1),axis=3)
    centEField = np.copy(Ex)
    backCentEField = np.copy(Ex)
    upperCentEField = np.copy(Ex)
    upperBackCentEField = np.copy(Ex)
    if BcZ == 0:
        centEField = np.append(centEField,centEField[:,:,centEField.shape[2]-1].reshape(centEField.shape[0],-1,1),axis=2)
        backCentEField = np.append(backCentEField[:,:,0].reshape(backCentEField.shape[0],-1,1),backCentEField,axis=2)
        upperCentEField = np.append(upperCentEField,
                                    upperCentEField[:,:,upperCentEField.shape[2]-1].reshape(upperCentEField.shape[0],-1,1),
                                    axis=2)
        upperBackCentEField = np.append(upperBackCentEField[:,:,0].reshape(upperBackCentEField.shape[0],-1,1),
                                        upperBackCentEField,axis=2)
    else:
        centEField = np.append(centEField,centEField[:,:,0].reshape(centEField.shape[0],-1,1),axis=2)
        backCentEField = np.append(backCentEField[:,:,backCentEField.shape[2]-1].reshape(backCentEField.shape[0],-1,1),
                                   backCentEField,axis=2)
        upperCentEField = np.append(upperCentEField,
                                    upperCentEField[:,:,0].reshape(upperCentEField.shape[0],-1,1),
                                    axis=2)
        upperBackCentEField = np.append(upperBackCentEField[:,:,upperBackCentEField.shape[2]-1].reshape(upperBackCentEField.shape[0],-1,1),
                                        upperBackCentEField,axis=2)
    if BcY == 0:
        centEField = np.append(centEField,
                               centEField[centEField.shape[0]-1,:,:].reshape(1,-1,centEField.shape[2]),axis=0)
        backCentEField = np.append(backCentEField,backCentEField[backCentEField.shape[0]-1,:,:].reshape(1,-1,backCentEField.shape[2]),axis=0)
        upperCentEField = np.append(upperCentEField[0,:,:].reshape(1,upperCentEField.shape[1],-1),
                                    upperCentEField,
                                    axis=0)
        upperBackCentEField = np.append(upperBackCentEField[0,:,:].reshape(1,-1,upperBackCentEField.shape[2]),
                                        upperBackCentEField,axis=0)
    else:
        centEField = np.append(centEField,
                               centEField[0,:,:].reshape(1,-1,centEField.shape[2]),axis=0)
        backCentEField = np.append(backCentEField,
                                   backCentEField[0,:,:].reshape(1,-1,backCentEField.shape[2]),
                                   axis=0)
        upperCentEField = np.append(upperCentEField[upperCentEField.shape[0]-1,:,:].reshape(1,upperCentEField.shape[1],-1),
                                    upperCentEField,
                                    axis=0)
        upperBackCentEField = np.append(upperBackCentEField[upperBackCentEField.shape[0]-1,:,:].reshape(1,-1,upperBackCentEField.shape[2]),
                                        upperBackCentEField,axis=0)
    if BcX == 0:
        centEField = np.append(centEField,
                               centEField[:,centEField.shape[1]-1,:].reshape(-1,1,centEField.shape[2]),axis=1)
        backCentEField = np.append(backCentEField,backCentEField[:,backCentEField.shape[1]-1,:].reshape(-1,1,backCentEField.shape[2]),axis=1)
        upperCentEField = np.append(upperCentEField,
                                    upperCentEField[:,upperCentEField.shape[1]-1,:].reshape(-1,1,upperCentEField.shape[2]),
                                    axis=1)
        upperBackCentEField = np.append(upperBackCentEField,
                                        upperBackCentEField[:,upperBackCentEField.shape[1]-1,:].reshape(-1,1,upperBackCentEField.shape[2]),
                                        axis=1)
    else:
        centEField = np.append(centEField,
                               centEField[:,0,:].reshape(-1,1,centEField.shape[2]),axis=1)
        backCentEField = np.append(backCentEField,backCentEField[:,0,:].reshape(-1,1,backCentEField.shape[2]),axis=1)
        upperCentEField = np.append(upperCentEField,
                                    upperCentEField[:,0,:].reshape(-1,1,upperCentEField.shape[2]),
                                    axis=1)
        upperBackCentEField = np.append(upperBackCentEField,
                                        upperBackCentEField[:,0,:].reshape(-1,1,upperBackCentEField.shape[2]),
                                        axis=1)

    #Calculate a term that is proportional to the change in Ex
    #from the back interface (k - 1/2) to the centre of cell k - 1
    zTerm1 = np.zeros(shape=(centEField.shape[0], centEField.shape[1], centEField.shape[2]))
    zTerm1[backTopFlux[0] > 0.0] = (upperBackFlux[5] - upperBackCentEField)[backTopFlux[0] > 0.0]
    zTerm1[backTopFlux[0] < 0.0] = (backFlux[5] - backCentEField)[backTopFlux[0] < 0.0]
    zTerm1[backTopFlux[0] == 0.0] = 0.5*(upperBackFlux[5] + backFlux[5]
                                         - upperBackCentEField - backCentEField)[backTopFlux[0] == 0.0]

    #Calculate a term that is proportional to the change in Ex
    #from the back interface (k - 1/2) to the centre of cell k
    zTerm2 = np.zeros(shape=(centEField.shape[0], centEField.shape[1], centEField.shape[2]))
    zTerm2[topFlux[0] > 0.0] = (upperBackFlux[5] - upperCentEField)[topFlux[0] > 0.0]
    zTerm2[topFlux[0] < 0.0] = (backFlux[5] - centEField)[topFlux[0] < 0.0]
    zTerm2[topFlux[0] == 0.0] = 0.5*(upperBackFlux[5] + backFlux[5]
                                     - upperCentEField - centEField)[topFlux[0] == 0.0]

    #Calculate a term that is proportional to the change in Ex
    #from the top interface (j - 1/2) to the centre of cell j-1
    yTerm1 = np.zeros(shape=(centEField.shape[0], centEField.shape[1], centEField.shape[2]))
    yTerm1[upperBackFlux[0] > 0.0] = (-backTopFlux[4] - upperBackCentEField)[upperBackFlux[0] > 0.0]
    yTerm1[upperBackFlux[0] < 0.0] = (-topFlux[4] - upperCentEField)[upperBackFlux[0] < 0.0]
    yTerm1[upperBackFlux[0] == 0.0] = 0.5*(-backTopFlux[4] - topFlux[4]
                                           - upperBackCentEField - upperCentEField)[upperBackFlux[0] == 0.0]

    #Calculate a term that is proportional to the change in Ex
    #from the top interface (j - 1/2) to the centre of cell j
    yTerm2 = np.zeros(shape=(centEField.shape[0], centEField.shape[1], centEField.shape[2]))
    yTerm2[backFlux[0] > 0.0] = (-backTopFlux[4] - backCentEField)[backFlux[0] > 0.0]
    yTerm2[backFlux[0] < 0.0] = (-topFlux[4] - centEField)[backFlux[0] < 0.0]
    yTerm2[backFlux[0] == 0.0] = 0.5*(-backTopFlux[4] - topFlux[4]
                                      - backCentEField - centEField)[backFlux[0] == 0.0]

    #Take the average of the four terms to get the Ex value at each cell corner
    cornerEMF = 0.25*(backFlux[5] + upperBackFlux[5] - topFlux[4] - backTopFlux[4]
                      + yTerm1 + yTerm2 + zTerm1 + zTerm2)
    return cornerEMF

######3D CORNER EMF FUNCTION (Y-COMPONENT)######

#Function: getYCornerEMF_3D
#Purpose: Calculates the Y-component of the electric field/EMF at every cell corner
#         in a 3D simulation.
#Input Parameters: interCellFluxX (the intercell flux vectors in the X-direction)
#                  interCellFluxZ (the intercell flux vectors in the z-direction)
#                  EY (the Y-component of the electric field at the centre
#                      of each cell)
#                  BcX (integer for the boundary condition in the x-direction
#                      (0 = outflow and 1 = periodic))
#                  BcY (integer for the boundary condition in the x-direction
#                      (0 = outflow and 1 = periodic))
#                  BcZ (integer for the boundary condition in the z-direction
#                      (0 = outflow and 1 = periodic))
#Outputs: cornerEMF (a matrix with the Y-component of the EMF at every cell corner in the simulation grid)
def getYCornerEMF_3D(interCellFluxX, interCellFluxZ, Ey, BcX, BcY, BcZ):
    backFlux = np.copy(interCellFluxZ)
    if BcY == 0:
        backFlux = np.append(backFlux,
                             backFlux[:,backFlux.shape[1]-1,:,:].reshape(7,1,-1,backFlux.shape[3]),
                             axis=1)
    else:
        backFlux = np.append(backFlux,
                             backFlux[:,0,:,:].reshape(7,1,-1,backFlux.shape[3]),
                             axis=1)
    if BcX == 0:
        leftBackFlux = np.append(backFlux[:,:,0,:].reshape(7,backFlux.shape[1],1,-1),backFlux,axis=2)
        backFlux = np.append(backFlux,backFlux[:,:,backFlux.shape[2]-1,:].reshape(7,backFlux.shape[1],1,-1),axis=2)
    else:
        leftBackFlux = np.append(backFlux[:,:,backFlux.shape[2]-1,:].reshape(7,backFlux.shape[1],1,-1),backFlux,axis=2)
        backFlux = np.append(backFlux,backFlux[:,:,0,:].reshape(7,backFlux.shape[1],1,-1),axis=2)
    leftFlux = np.copy(interCellFluxX)
    if BcY == 0:
        leftFlux = np.append(leftFlux,leftFlux[:,leftFlux.shape[1]-1,:,:].reshape(7,1,leftFlux.shape[2],-1),axis=1)
    else:
        leftFlux = np.append(leftFlux,leftFlux[:,0,:,:].reshape(7,1,leftFlux.shape[2],-1),axis=1)
    if BcZ == 0:
        backLeftFlux = np.append(leftFlux[:,:,:,0].reshape(7,leftFlux.shape[1],-1,1),leftFlux,axis=3)
        leftFlux = np.append(leftFlux,leftFlux[:,:,:,leftFlux.shape[3]-1].reshape(7,leftFlux.shape[1],-1,1),axis=3)
    else:
        backLeftFlux = np.append(leftFlux[:,:,:,leftFlux.shape[3]-1].reshape(7,leftFlux.shape[1],-1,1),leftFlux,axis=3)
        leftFlux = np.append(leftFlux,leftFlux[:,:,:,0].reshape(7,leftFlux.shape[1],-1,1),axis=3)
    centEField = np.copy(Ey)
    backCentEField = np.copy(Ey)
    leftCentEField = np.copy(Ey)
    backLeftCentEField = np.copy(Ey)
    if BcZ == 0:
        centEField = np.append(centEField,centEField[:,:,centEField.shape[2]-1].reshape(centEField.shape[0],-1,1),axis=2)
        backCentEField = np.append(backCentEField[:,:,0].reshape(backCentEField.shape[0],-1,1),backCentEField,axis=2)
        leftCentEField = np.append(leftCentEField,
                                    leftCentEField[:,:,leftCentEField.shape[2]-1].reshape(leftCentEField.shape[0],-1,1),
                                    axis=2)
        backLeftCentEField = np.append(backLeftCentEField[:,:,0].reshape(backLeftCentEField.shape[0],-1,1),
                                        backLeftCentEField,axis=2)
    else:
        centEField = np.append(centEField,centEField[:,:,0].reshape(centEField.shape[0],-1,1),axis=2)
        backCentEField = np.append(backCentEField[:,:,backCentEField.shape[2]-1].reshape(backCentEField.shape[0],-1,1),
                                   backCentEField,axis=2)
        leftCentEField = np.append(leftCentEField,
                                    leftCentEField[:,:,0].reshape(leftCentEField.shape[0],-1,1),
                                    axis=2)
        backLeftCentEField = np.append(backLeftCentEField[:,:,backLeftCentEField.shape[2]-1].reshape(backLeftCentEField.shape[0],-1,1),
                                        backLeftCentEField,axis=2)
    if BcY == 0:
        centEField = np.append(centEField,
                               centEField[centEField.shape[0]-1,:,:].reshape(1,-1,centEField.shape[2]),axis=0)
        backCentEField = np.append(backCentEField,backCentEField[backCentEField.shape[0]-1,:,:].reshape(1,-1,backCentEField.shape[2]),axis=0)
        leftCentEField = np.append(leftCentEField,
                                   leftCentEField[leftCentEField.shape[0]-1, :, :].reshape(1, leftCentEField.shape[1], -1),
                                    axis=0)
        backLeftCentEField = np.append(backLeftCentEField,
                                   backLeftCentEField[backLeftCentEField.shape[0]-1, :, :].reshape(1, backLeftCentEField.shape[1], -1),
                                    axis=0)
    else:
        centEField = np.append(centEField,
                               centEField[0,:,:].reshape(1,-1,centEField.shape[2]),axis=0)
        backCentEField = np.append(backCentEField,backCentEField[0,:,:].reshape(1,-1,backCentEField.shape[2]),axis=0)
        leftCentEField = np.append(leftCentEField,
                                   leftCentEField[0, :, :].reshape(1, leftCentEField.shape[1], -1),
                                    axis=0)
        backLeftCentEField = np.append(backLeftCentEField,
                                   backLeftCentEField[0, :, :].reshape(1, backLeftCentEField.shape[1], -1),
                                    axis=0)
    if BcX == 0:
        centEField = np.append(centEField,
                               centEField[:,centEField.shape[1]-1,:].reshape(-1,1,centEField.shape[2]),axis=1)
        backCentEField = np.append(backCentEField,backCentEField[:,backCentEField.shape[1]-1,:].reshape(-1,1,backCentEField.shape[2]),axis=1)
        leftCentEField = np.append(leftCentEField[:,0,:].reshape(-1,1,leftCentEField.shape[2]),
                                   leftCentEField,
                                    axis=1)
        backLeftCentEField = np.append(backLeftCentEField[:,0,:].reshape(-1,1,backLeftCentEField.shape[2]),
                                   backLeftCentEField,
                                    axis=1)
    else:
        centEField = np.append(centEField,
                               centEField[:,0,:].reshape(-1,1,centEField.shape[2]),axis=1)
        backCentEField = np.append(backCentEField,backCentEField[:,0,:].reshape(-1,1,backCentEField.shape[2]),axis=1)
        leftCentEField = np.append(leftCentEField[:,leftCentEField.shape[1]-1,:].reshape(-1,1,leftCentEField.shape[2]),
                                   leftCentEField,
                                    axis=1)
        backLeftCentEField = np.append(backLeftCentEField[:,backLeftCentEField.shape[1]-1,:].reshape(-1,1,backLeftCentEField.shape[2]),
                                   backLeftCentEField,
                                    axis=1)

    #Calculate a term that is proportional to the change in Ey
    #from the back interface (k - 1/2) to the centre of cell k - 1
    zTerm1 = np.zeros(shape=(centEField.shape[0], centEField.shape[1], centEField.shape[2]))
    zTerm1[backLeftFlux[0] > 0.0] = (-leftBackFlux[4] - backLeftCentEField)[backLeftFlux[0] > 0.0]
    zTerm1[backLeftFlux[0] < 0.0] = (-backFlux[4] - backCentEField)[backLeftFlux[0] < 0.0]
    zTerm1[backLeftFlux[0] == 0.0] = 0.5*(-leftBackFlux[4] - backFlux[4]
                                          - backLeftCentEField - backCentEField)[backLeftFlux[0] == 0.0]

    #Calculate a term that is proportional to the change in Ey
    #from the back interface (k - 1/2) to the centre of cell k
    zTerm2 = np.zeros(shape=(centEField.shape[0], centEField.shape[1], centEField.shape[2]))
    zTerm2[leftFlux[0] > 0.0] = (-leftBackFlux[4] - leftCentEField)[leftFlux[0] > 0.0]
    zTerm2[leftFlux[0] < 0.0] = (-backFlux[4] - centEField)[leftFlux[0] < 0.0]
    zTerm2[leftFlux[0] == 0.0] = 0.5*(-leftBackFlux[4] - backFlux[4]
                                      - leftCentEField - centEField)[leftFlux[0] == 0.0]

    #Calculate a term that is proportional to the change in Ey
    #from the back interface (i - 1/2) to the centre of cell i - 1
    xTerm1 = np.zeros(shape=(centEField.shape[0], centEField.shape[1], centEField.shape[2]))
    xTerm1[leftBackFlux[0] > 0.0] = (backLeftFlux[5] - backLeftCentEField)[leftBackFlux[0] > 0.0]
    xTerm1[leftBackFlux[0] < 0.0] = (leftFlux[5] - leftCentEField)[leftBackFlux[0] < 0.0]
    xTerm1[leftBackFlux[0] == 0.0] = 0.5*(backLeftFlux[5] + leftFlux[5]
                                          - backLeftCentEField - leftCentEField)[leftBackFlux[0] == 0.0]

    #Calculate a term that is proportional to the change in Ey
    #from the back interface (i - 1/2) to the centre of cell i
    xTerm2 = np.zeros(shape=(centEField.shape[0], centEField.shape[1], centEField.shape[2]))
    xTerm2[backFlux[0] > 0.0] = (backLeftFlux[5] - backCentEField)[backFlux[0] > 0.0]
    xTerm2[backFlux[0] < 0.0] = (leftFlux[5] - centEField)[backFlux[0] < 0.0]
    xTerm2[backFlux[0] == 0.0] = 0.5*(backLeftFlux[5] + leftFlux[5]
                                      - backCentEField - centEField)[backFlux[0] == 0.0]

    #Take the average of the four terms to get the Ey value at each cell corner
    cornerEMF = 0.25 * (leftFlux[5] + backLeftFlux[5] - backFlux[4] - leftBackFlux[4]
                        + xTerm1 + xTerm2 + zTerm1 + zTerm2)
    return cornerEMF