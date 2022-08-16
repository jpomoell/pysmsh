# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Centered finite difference operations
"""

import numpy as np
import numba

from pysmsh.core.colocation import ColocationId


@numba.njit()
def gradient(scalar, vec):

    if scalar.coloc.typeid != ColocationId.cell_centered:
        raise ValueError("Input scalar field should be cell centered")

    if vec.coloc.typeid != ColocationId.cell_centered:
        raise ValueError("Output vector field should be cell centered")

    if not np.all(vec.mesh.num_cells == scalar.mesh.num_cells):
        raise ValueError("Fields with different number of cells given as inputs")

    u = scalar.data

    # Cell-center coordinates
    x, y, z = scalar.mesh.centers
    
    # Number of cells in total
    num_cells = scalar.mesh.num_cells
    
    # Components of the resulting gradient    
    vx, vy, vz = vec.data

    #
    # grad(u)_x  @  (x_{i}, y_{j}, z_{k})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                vx[i, j, k] = (u[i+1, j, k] - u[i-1, j, k])/(x[i+1] - x[i-1])
    
    #
    # grad(u)_y  @  (x_{i}, y_{j}, z_{k})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                vy[i, j, k] = (u[i, j+1, k] - u[i, j-1, k])/(y[j+1] - y[j-1])

    #
    # grad(u)_z  @  (x_{i}, y_{j}, z_{k})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                vz[i, j, k] = (u[i, j, k+1] - u[i, j, k-1])/(z[k+1] - z[k-1])



@numba.njit()
def divergence(vec, scalar):

    if vec.coloc.typeid != ColocationId.cell_centered:
        raise ValueError("Input vector field should be cell centered")

    if scalar.coloc.typeid != ColocationId.cell_centered:
        raise ValueError("Output scalar field should be cell centered")

    if not np.all(vec.mesh.num_cells == scalar.mesh.num_cells):
        raise ValueError("Fields with different number of cells given as inputs")
    
    
    div_u = scalar.data

    # Number of cells in total
    num_cells = vec.mesh.num_cells
    
    # Components of the input vector field
    vx, vy, vz = vec.data

    # Cell sizes
    dx, dy, dz = vec.mesh.spacing

    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                div_u[i, j, k] \
                    =  (vx[i+1, j, k] - vx[i-1, j, k])/(2.0*dx[i]) \
                     + (vy[i, j+1, k] - vy[i, j-1, k])/(2.0*dy[j]) \
                     + (vz[i, j, k+1] - vz[i, j, k-1])/(2.0*dz[k])
    

@numba.njit()
def curl(invec, outvec):

    if invec.coloc.typeid != ColocationId.cell_centered:
        raise ValueError("Input vector field should be cell centered")

    if outvec.coloc.typeid != ColocationId.cell_centered:
        raise ValueError("Output vector field should be cell centered")

    if not np.all(invec.mesh.num_cells == outvec.mesh.num_cells):
        raise ValueError("Fields with different number of cells given as inputs")
            
    # Number of cells in total
    num_cells = invec.mesh.num_cells
    
    # Components of the input vector field
    vx, vy, vz = invec.data

    # Components of the output vector field
    curlv_x, curlv_y, curlv_z = outvec.data

    # Cell sizes
    dx, dy, dz = invec.mesh.spacing

    #
    # curl(v)_x  @  (x_{i}, y_{j}, z_{k})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                curlv_x[i, j, k] \
                    =  (vz[i, j+1, k] - vz[i, j-1, k])/(2.0*dy[j]) \
                     - (vy[i, j, k+1] - vy[i, j, k-1])/(2.0*dz[k]) 
                     
    #
    # curl(v)_y  @  (x_{i}, y_{j}, z_{k})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                curlv_y[i, j, k] \
                    =  (vx[i, j, k+1] - vx[i, j, k-1])/(2.0*dz[k]) \
                     - (vz[i+1, j, k] - vz[i-1, j, k])/(2.0*dx[i])
   
    #
    # curl(v)_z  @  (x_{i}, y_{j}, z_{k})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                curlv_z[i, j, k] \
                    =  (vy[i+1, j, k] - vy[i-1, j, k])/(2.0*dx[i]) \
                     - (vx[i, j+1, k] - vx[i, j-1, k])/(2.0*dy[j])
   