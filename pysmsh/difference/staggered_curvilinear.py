# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Staggered finite difference operations on orthogonal curvilinear domains
"""

import numpy as np
import numba
import numba.experimental.jitclass as jitclass

from pysmsh.core.colocation import ColocationId


@jitclass()
class SphericalGeometry:
    
    def __init__(self):
        pass
    
    def h1(self, r, t, p):
        return 1.0

    def h2(self, r, t, p):
        return r

    def h3(self, r, t, p):
        return r*np.sin(t)


@jitclass()
class ToroidalGeometry:
    
    a: float
    
    def __init__(self, a=1.0):
        self.a = a
    
    def h1(self, nu, u, phi):
        return self.a/np.abs(np.cosh(nu)-np.cos(u))

    def h2(self, nu, u, phi):
        return self.a/np.abs(np.cosh(nu)-np.cos(u))

    def h3(self, nu, u, phi):
        return self.a*np.sinh(nu)/np.abs(np.cosh(nu)-np.cos(u))


@jitclass()
class ToroidallyCurvedCylindricalGeometry:
    
    R: float
    
    def __init__(self, major_radius=1.0):
        self.R = major_radius
    
    def h1(self, r, phi, theta):
        return 1.0
        
    def h2(self, r, phi, theta):
        return self.R + r*np.cos(theta)

    def h3(self, r, phi, theta):
        return r


@numba.njit(boundscheck=True)
def gradient(scalar, vec, geometry):

    if scalar.coloc.typeid != ColocationId.cell_centered:
        raise ValueError("Input scalar field should be cell centered")

    if vec.coloc.typeid != ColocationId.face_staggered:
        raise ValueError("Output vector field should be face staggered")

    if not np.all(vec.mesh.num_cells == scalar.mesh.num_cells):
        raise ValueError("Fields with different number of cells given as inputs")

    u = scalar.data

    # Cell-center coordinates
    x1, x2, x3 = scalar.mesh.centers

    # Mesh spacings
    dx1, dx2, dx3 = scalar.mesh.spacing

    # Number of cells in total
    num_cells = scalar.mesh.num_cells
    
    # Components of the resulting gradient field   
    v1, v2, v3 = vec.data

    #
    # grad(u)_{1; i-1/2, j, k} = (u_{i, j, k} - u_{i-1, j, k})/( dx_1 h_1{i-1/2, j, k}) )
    #
    for i in range(1, num_cells[0]):
        for j in range(0, num_cells[1]):
            for k in range(0, num_cells[2]):

                h1 = geometry.h1(x1[i] - 0.5*dx1[i], x2[j], x3[k])
                d1 = x1[i] - x1[i-1]

                v1[i, j, k] = (u[i, j, k] - u[i-1, j, k])/(d1*h1)
    
    #
    # grad(u)_{y; i, j-1/2, k} = (u_{i, j, k} - u_{i, j-1, k})/dy
    #
    for i in range(0, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(0, num_cells[2]):
                
                h2 = geometry.h2(x1[i], x2[j] - 0.5*dx2[j], x3[k])
                d2 = x2[j] - x2[j-1]

                v2[i, j, k] = (u[i, j, k] - u[i, j-1, k])/(d2*h2)

    #
    # grad(u)_{z; i, j, k-1/2} = (u_{i, j, k} - u_{i, j, k-1})/dz
    #
    for i in range(0, num_cells[0]):
        for j in range(0, num_cells[1]):
            for k in range(1, num_cells[2]):

                h3 = geometry.h3(x1[i], x2[j], x3[k] - 0.5*dx3[k])
                d3 = x3[k] - x3[k-1]

                v3[i, j, k] = (u[i, j, k] - u[i, j, k-1])/(d3*h3)


@numba.njit()
def div_face(vec, scalar, geometry):

    if vec.coloc.typeid != ColocationId.face_staggered:
        raise ValueError("Input vector field should be face staggered")

    if scalar.coloc.typeid != ColocationId.cell_centered:
        raise ValueError("Output scalar field should be cell centered")

    if not np.all(vec.mesh.num_cells == scalar.mesh.num_cells):
        raise ValueError("Fields with different number of cells given as inputs")
    
    
    div_u = scalar.data

    # Number of cells in total
    num_cells = vec.mesh.num_cells
    
    # Components of the input vector field
    v1, v2, v3 = vec.data

    # Mesh edge coordinates
    x1e, x2e, x3e = vec.mesh.edges
    
    # Mesh center coordinates
    x1c, x2c, x3c = vec.mesh.centers

    # Cell sizes
    dx1, dx2, dx3 = vec.mesh.spacing

    for i in range(0, num_cells[0]):
        for j in range(0, num_cells[1]):
            for k in range(0, num_cells[2]):

                diff_x1 \
                    = geometry.h2(x1e[i+1], x2c[j], x3c[k])*geometry.h3(x1e[i+1], x2c[j], x3c[k])*v1[i+1, j, k] \
                    - geometry.h2(x1e[i  ], x2c[j], x3c[k])*geometry.h3(x1e[i  ], x2c[j], x3c[k])*v1[i  , j, k]

                diff_x2 \
                    = geometry.h1(x1c[i], x2e[j+1], x3c[k])*geometry.h3(x1c[i], x2e[j+1], x3c[k])*v2[i, j+1, k] \
                    - geometry.h1(x1c[i], x2e[j  ], x3c[k])*geometry.h3(x1c[i], x2e[j  ], x3c[k])*v2[i, j  , k]

                diff_x3 \
                    = geometry.h1(x1c[i], x2c[j], x3e[k+1])*geometry.h2(x1c[i], x2c[j], x3e[k+1])*v3[i, j, k+1] \
                    - geometry.h1(x1c[i], x2c[j], x3e[k  ])*geometry.h2(x1c[i], x2c[j], x3e[k  ])*v3[i, j, k  ]

                h123 \
                    = geometry.h1(x1c[i], x2c[j], x3c[k]) \
                    * geometry.h2(x1c[i], x2c[j], x3c[k]) \
                    * geometry.h3(x1c[i], x2c[j], x3c[k])

                div_u[i, j, k] \
                    = (diff_x1/dx1[i] + diff_x2/dx2[j] + diff_x3/dx3[k])/h123
    

@numba.njit(boundscheck=True, error_model="numpy")
def div_edge(vec, scalar, geometry):

    if vec.coloc.typeid != ColocationId.edge_staggered:
        raise ValueError("Input vector field should be edge staggered")

    if scalar.coloc.typeid != ColocationId.node_centered:
        raise ValueError("Output scalar field should be node centered")

    if not np.all(vec.mesh.num_cells == scalar.mesh.num_cells):
        raise ValueError("Fields with different number of cells given as inputs")
    
    div_u = scalar.data

    # Number of cells in total
    num_cells = vec.mesh.num_cells
    
    # Components of the input vector field
    v1, v2, v3 = vec.data

    # Mesh edge coordinates
    x1e, x2e, x3e = vec.mesh.edges
    
    # Mesh center coordinates
    x1c, x2c, x3c = vec.mesh.centers

    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):

                diff_x1 \
                    = geometry.h2(x1c[i  ], x2e[j], x3e[k])*geometry.h3(x1c[i  ], x2e[j], x3e[k])*v1[i  , j, k] \
                    - geometry.h2(x1c[i-1], x2e[j], x3e[k])*geometry.h3(x1c[i-1], x2e[j], x3e[k])*v1[i-1, j, k]

                diff_x2 \
                    = geometry.h1(x1e[i], x2c[j  ], x3e[k])*geometry.h3(x1e[i], x2c[j  ], x3e[k])*v2[i, j  , k] \
                    - geometry.h1(x1e[i], x2c[j-1], x3e[k])*geometry.h3(x1e[i], x2c[j-1], x3e[k])*v2[i, j-1, k]

                diff_x3 \
                    = geometry.h1(x1e[i], x2e[j], x3c[k  ])*geometry.h2(x1e[i], x2e[j], x3c[k  ])*v3[i, j, k  ] \
                    - geometry.h1(x1e[i], x2e[j], x3c[k-1])*geometry.h2(x1e[i], x2e[j], x3c[k-1])*v3[i, j, k-1]

                h123 \
                    = geometry.h1(x1e[i], x2e[j], x3e[k]) \
                    * geometry.h2(x1e[i], x2e[j], x3e[k]) \
                    * geometry.h3(x1e[i], x2e[j], x3e[k])

                dx1 = x1c[i] - x1c[i-1]
                dx2 = x2c[j] - x2c[j-1]
                dx3 = x3c[k] - x3c[k-1]
                
                div_u[i, j, k] = (diff_x1/dx1 + diff_x2/dx2 + diff_x3/dx3)/h123


@numba.njit(boundscheck=True, error_model="numpy")
def divergence(vec, scalar, geometry):

    if vec.coloc.typeid == ColocationId.edge_staggered:
        div_edge(vec, scalar, geometry)
    elif vec.coloc.typeid == ColocationId.face_staggered:
        div_face(vec, scalar, geometry)
    else:
        raise ValueError("Input vector field should be either face or edge staggered")


@numba.njit(boundscheck=True, error_model="numpy")
def curl_edge(invec, outvec, geometry):

    if invec.coloc.typeid != ColocationId.edge_staggered:
        raise ValueError("Input vector field should be edge staggered")

    if outvec.coloc.typeid != ColocationId.face_staggered:
        raise ValueError("Output vector field should be face staggered")

    if not np.all(invec.mesh.num_cells == outvec.mesh.num_cells):
        raise ValueError("Fields with different number of cells given as inputs")
            
    # Number of cells in total
    num_cells = invec.mesh.num_cells
    
    # Components of the input vector field
    v1, v2, v3 = invec.data

    # Components of the output vector field
    curlv_1, curlv_2, curlv_3 = outvec.data

    # Mesh edge coordinates
    x1e, x2e, x3e = invec.mesh.edges
    
    # Mesh center coordinates
    x1c, x2c, x3c = invec.mesh.centers

    # Cell sizes
    dx1, dx2, dx3 = invec.mesh.spacing

    #
    # curl(v)_1  @  (x1_{i-1/2}, \x2_{j}, \x3_{k})
    #
    for i in range(0, num_cells[0] + 1):
        for j in range(0, num_cells[1]):
            for k in range(0, num_cells[2]):

                diff_x2 \
                    = (   geometry.h3(x1e[i], x2e[j+1], x3c[k])*v3[i, j+1, k] \
                        - geometry.h3(x1e[i], x2e[j  ], x3c[k])*v3[i, j  , k] )/dx2[j]

                diff_x3 \
                    = (   geometry.h2(x1e[i], x2c[j], x3e[k+1])*v2[i, j, k+1] \
                        - geometry.h2(x1e[i], x2c[j], x3e[k  ])*v2[i, j, k  ] )/dx3[k]

                curlv_1[i, j, k] \
                    = (diff_x2 - diff_x3)/(geometry.h2(x1e[i], x2c[j], x3c[k])*geometry.h3(x1e[i], x2c[j], x3c[k]))

    #
    # curl(v)_2  @  (x1_{i}, \x2_{j-1/2}, \x3_{k})
    #
    for i in range(0, num_cells[0]):
        for j in range(0, num_cells[1] + 1):
            for k in range(0, num_cells[2]):

                diff_x3 \
                    = (   geometry.h1(x1c[i], x2e[j], x3e[k+1])*v1[i, j, k+1] \
                        - geometry.h1(x1c[i], x2e[j], x3e[k  ])*v1[i, j, k  ] )/dx3[k]

                diff_x1 \
                    = (   geometry.h3(x1e[i+1], x2e[j], x3c[k])*v3[i+1, j, k] \
                        - geometry.h3(x1e[i  ], x2e[j], x3c[k])*v3[i  , j, k] )/dx1[i]

                curlv_2[i, j, k] \
                    = (diff_x3 - diff_x1)/(geometry.h1(x1c[i], x2e[j], x3c[k])*geometry.h3(x1c[i], x2e[j], x3c[k]))
   
    #
    # curl(v)_3 @  (x1_{i}, \x2_{j}, \x3_{k-1/2})
    #
    for i in range(0, num_cells[0]):
        for j in range(0, num_cells[1]):
            for k in range(0, num_cells[2] + 1):

                diff_x1 \
                    = (   geometry.h2(x1e[i+1], x2c[j], x3e[k])*v2[i+1, j, k] \
                        - geometry.h2(x1e[i  ], x2c[j], x3e[k])*v2[i  , j, k] )/dx1[i]

                diff_x2 \
                    = (   geometry.h1(x1c[i], x2e[j+1], x3e[k])*v1[i, j+1, k] \
                        - geometry.h1(x1c[i], x2e[j  ], x3e[k])*v1[i, j  , k] )/dx2[j]

                curlv_3[i, j, k] \
                    = (diff_x1 - diff_x2)/(geometry.h1(x1c[i], x2c[j], x3e[k])*geometry.h2(x1c[i], x2c[j], x3e[k]))


@numba.njit(boundscheck=True)
def curl_face(invec, outvec, geometry):

    # Note: on non-uniform grids, accruacy may not be as expected

    if invec.coloc.typeid != ColocationId.face_staggered:
        raise ValueError("Input vector field should be face staggered")

    if outvec.coloc.typeid != ColocationId.edge_staggered:
        raise ValueError("Output vector field should be edge staggered")

    if not np.all(invec.mesh.num_cells == outvec.mesh.num_cells):
        raise ValueError("Fields with different number of cells given as inputs")
            
    # Number of cells in total
    num_cells = invec.mesh.num_cells
    
    # Components of the input vector field
    v1, v2, v3 = invec.data

    # Components of the output vector field
    curlv_1, curlv_2, curlv_3 = outvec.data

    # Mesh edge coordinates
    x1e, x2e, x3e = invec.mesh.edges
    
    # Mesh center coordinates
    x1c, x2c, x3c = invec.mesh.centers


    #
    # curl(v)_1  @  (x1_{i}, \x2_{j-1/2}, \x3_{k-1/2})
    #
    for i in range(0, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                
                dx2 = x2c[j] - x2c[j-1]
                dx3 = x3c[k] - x3c[k-1]

                diff_x2 \
                    = (   geometry.h3(x1c[i], x2c[j  ], x3e[k])*v3[i, j  , k] \
                        - geometry.h3(x1c[i], x2c[j-1], x3e[k])*v3[i, j-1, k] )/dx2

                diff_x3 \
                    = (   geometry.h2(x1c[i], x2e[j], x3c[k  ])*v2[i, j, k  ] \
                        - geometry.h2(x1c[i], x2e[j], x3c[k-1])*v2[i, j, k-1] )/dx3

                curlv_1[i, j, k] \
                    = (diff_x2 - diff_x3)/(geometry.h2(x1c[i], x2e[j], x3e[k])*geometry.h3(x1c[i], x2e[j], x3e[k]))


    #
    # curl(v)_2  @  (x1_{i-1/2}, \x2_{j}, \x3_{k-1/2})
    #
    for i in range(1, num_cells[0]):
        for j in range(0, num_cells[1]):
            for k in range(1, num_cells[2]):

                dx1 = x1c[i] - x1c[i-1]
                dx3 = x3c[k] - x3c[k-1]

                diff_x3 \
                    = (   geometry.h1(x1e[i], x2c[j], x3c[k  ])*v1[i, j, k  ] \
                        - geometry.h1(x1e[i], x2c[j], x3c[k-1])*v1[i, j, k-1] )/dx3

                diff_x1 \
                    = (   geometry.h3(x1c[i  ], x2c[j], x3e[k])*v3[i  , j, k] \
                        - geometry.h3(x1c[i-1], x2c[j], x3e[k])*v3[i-1, j, k] )/dx1

                curlv_2[i, j, k] \
                    = (diff_x3 - diff_x1)/(geometry.h1(x1e[i], x2c[j], x3e[k])*geometry.h3(x1e[i], x2c[j], x3e[k]))
   
   
    #
    # curl(v)_3 @  (x1_{i-1/2}, \x2_{j-1/2}, \x3_{k})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(0, num_cells[2]):
                
                dx1 = x1c[i] - x1c[i-1]
                dx2 = x2c[j] - x2c[j-1]

                diff_x1 \
                    = (   geometry.h2(x1c[i  ], x2e[j], x3c[k])*v2[i  , j, k] \
                        - geometry.h2(x1c[i-1], x2e[j], x3c[k])*v2[i-1, j, k] )/dx1

                diff_x2 \
                    = (   geometry.h1(x1e[i], x2c[j  ], x3c[k])*v1[i, j  , k] \
                        - geometry.h1(x1e[i], x2c[j-1], x3c[k])*v1[i, j-1, k] )/dx2

                curlv_3[i, j, k] \
                    = (diff_x1 - diff_x2)/(geometry.h1(x1e[i], x2e[j], x3c[k])*geometry.h2(x1e[i], x2e[j], x3c[k]))


@numba.njit(error_model="numpy")
def curl(invec, outvec, geometry):

    if invec.coloc.typeid == ColocationId.edge_staggered:
        curl_edge(invec, outvec, geometry)
    elif invec.coloc.typeid == ColocationId.face_staggered:
        curl_face(invec, outvec, geometry)
    else:
        raise ValueError("Input vector field should be either face or edge staggered")
