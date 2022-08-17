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
import numba.types
import numba.experimental.jitclass as jitclass

import pysmsh


@jitclass()
class TrilinearInterpolation:
    """Trilinear interpolation
    """

    # Coordinates of the data points
    coordinates: numba.types.UniTuple(numba.float64[:], 3)
    
    # Reference to field data
    data : numba.types.Array(numba.float64, 3, "C")
    
    def __init__(self, data, coordinates, copy_coordinates=True):        

        # Set the coordinates corresponding to the values on the grid
        if copy_coordinates:
            self.coordinates = (np.copy(coordinates[0]), np.copy(coordinates[1]), np.copy(coordinates[2]))
        else:
            self.coordinates = (coordinates[0], coordinates[1], coordinates[2])

        self.data = data

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
    
    def value(self, p):

        i, j, k = self.index(p)

        # Coordinate of point in local unit cube
        xu, yu, zu = self.unit_cube_coordinate(p)
                
        value =   (1.0-xu)*(1.0-yu)*(1.0-zu)*self.data[i  , j  , k  ] \
                + (    xu)*(1.0-yu)*(1.0-zu)*self.data[i+1, j  , k  ] \
                + (1.0-xu)*(    yu)*(1.0-zu)*self.data[i,   j+1, k  ] \
                + (    xu)*(    yu)*(1.0-zu)*self.data[i+1, j+1, k  ] \
                + (1.0-xu)*(1.0-yu)*(    zu)*self.data[i,   j  , k+1] \
                + (    xu)*(1.0-yu)*(    zu)*self.data[i+1, j  , k+1] \
                + (1.0-xu)*(    yu)*(    zu)*self.data[i  , j+1, k+1] \
                + (    xu)*(    yu)*(    zu)*self.data[i+1, j+1, k+1]

        return value




@jitclass()
class CenteredVectorInterpolator:
    
    # Domain extent
    indomain_extent: numba.types.UniTuple(numba.types.Array(numba.float64, 1, "C"), 3)

    components : numba.types.UniTuple(TrilinearInterpolation.class_type.instance_type, 3)
    
    def __init__(self, field):
        
        copy_coordinates = True

        v1_interp = TrilinearInterpolation(field.data[0], field.mesh.centers, copy_coordinates)
        v2_interp = TrilinearInterpolation(field.data[1], field.mesh.centers, copy_coordinates)
        v3_interp = TrilinearInterpolation(field.data[2], field.mesh.centers, copy_coordinates)
        
        self.indomain_extent = (field.mesh.indomain_extent[0], field.mesh.indomain_extent[1], field.mesh.indomain_extent[2])

        self.components = (v1_interp, v2_interp, v3_interp)

    def value(self, p):

        v1 = self.components[0].value(p)
        v2 = self.components[1].value(p)
        v3 = self.components[2].value(p)
        
        return np.array((v1, v2, v3))

    def cartesian_unit_vector(self, p):
        
        v = self.value(p)

        vsqr = v[0]**2 + v[1]**2 + v[2]**2

        if vsqr > 0.0:
            return v/np.sqrt(vsqr)
        else:
            return np.zeros(3)

    def cell_size(self, p):

        obj = self.components[0]

        i, j, k = obj.index(p)
        
        dx = obj.coordinates[0][i+1] - obj.coordinates[0][i]
        dy = obj.coordinates[1][j+1] - obj.coordinates[1][j]
        dz = obj.coordinates[2][k+1] - obj.coordinates[2][k]
        
        return np.array((dx, dy, dz))


@jitclass()
class EdgeStaggeredInterpolator:
    
    # Domain extent
    indomain_extent: numba.types.UniTuple(numba.types.Array(numba.float64, 1, "C"), 3)

    components : numba.types.UniTuple(TrilinearInterpolation.class_type.instance_type, 3)
    
    def __init__(self, field):
        
        copy_coordinates = True

        v1_interp = TrilinearInterpolation(field.data[0], field.mesh.edge_center_coords(0), copy_coordinates)
        v2_interp = TrilinearInterpolation(field.data[1], field.mesh.edge_center_coords(1), copy_coordinates)
        v3_interp = TrilinearInterpolation(field.data[2], field.mesh.edge_center_coords(2), copy_coordinates)
        
        self.indomain_extent = (field.mesh.indomain_extent[0], field.mesh.indomain_extent[1], field.mesh.indomain_extent[2])

        self.components = (v1_interp, v2_interp, v3_interp)

    def value(self, p):

        v1 = self.components[0].value(p)
        v2 = self.components[1].value(p)
        v3 = self.components[2].value(p)
        
        return np.array((v1, v2, v3))

    def cartesian_unit_vector(self, p):
        
        v = self.value(p)

        vsqr = v[0]**2 + v[1]**2 + v[2]**2

        if vsqr > 0.0:
            return v/np.sqrt(vsqr)
        else:
            return np.zeros(3)

    def cell_size(self, p):

        obj = self.components[0]

        i, j, k = obj.index(p)
        
        dx = obj.coordinates[0][i+1] - obj.coordinates[0][i]
        dy = obj.coordinates[1][j+1] - obj.coordinates[1][j]
        dz = obj.coordinates[2][k+1] - obj.coordinates[2][k]
        
        return np.array((dx, dy, dz))
