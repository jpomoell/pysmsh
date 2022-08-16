# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Trilinear interpolation
"""

import numpy as np
import numba
import numba.experimental.jitclass as jitclass


@jitclass()
class TrilinearInterpolation:
    """Trilinear interpolation
    """

    coordinates: numba.types.UniTuple(numba.float64[:], 3)
    
    def __init__(self):        
        pass

    def set_coordinates(self, x, y, z, copy=True):
        """Set the coordinates corresponding to the values on the grid
        """

        if copy:
            self.coordinates = (np.copy(x), np.copy(y), np.copy(z))
        else:
            self.coordinates = (x, y, z)
            
    def index(self, p):

        i = np.searchsorted(self.coordinates[0], p[0]) - 1 
        j = np.searchsorted(self.coordinates[1], p[1]) - 1
        k = np.searchsorted(self.coordinates[2], p[2]) - 1

        return i, j, k
   
    def unit_cube_coordinate(self, p):
        
        i, j, k = self.index(p)
        
        xi, yj, zk = self.coordinates[0][i], self.coordinates[1][j], self.coordinates[2][k]
        
        dx = self.coordinates[0][i+1] - xi
        dy = self.coordinates[1][j+1] - yj
        dz = self.coordinates[2][k+1] - zk
        
        return (p[0]-xi)/dx, (p[1] - yj)/dy, (p[2] - zk)/dz
    
    def value(self, p, data):

        i, j, k = self.index(p)

        # Coordinate of point in local unit cube
        xu, yu, zu = self.unit_cube_coordinate(p)
                
        value =   (1.0-xu)*(1.0-yu)*(1.0-zu)*data[i  , j  , k  ] \
                + (    xu)*(1.0-yu)*(1.0-zu)*data[i+1, j  , k  ] \
                + (1.0-xu)*(    yu)*(1.0-zu)*data[i,   j+1, k  ] \
                + (    xu)*(    yu)*(1.0-zu)*data[i+1, j+1, k  ] \
                + (1.0-xu)*(1.0-yu)*(    zu)*data[i,   j  , k+1] \
                + (    xu)*(1.0-yu)*(    zu)*data[i+1, j  , k+1] \
                + (1.0-xu)*(    yu)*(    zu)*data[i  , j+1, k+1] \
                + (    xu)*(    yu)*(    zu)*data[i+1, j+1, k+1]

        return value