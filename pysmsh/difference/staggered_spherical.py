# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Staggered finite difference operations on spherical domains
"""

import numpy as np
import numba

from pysmsh.core.colocation import ColocationId


# TODO: index ranges
# TODO: ghost cells

@numba.njit()
def gradient(scalar, vec):

    raise ValueError("Not implemented")
    
    if scalar.coloc.typeid != ColocationId.cell_centered:
        raise ValueError("Input scalar field should be cell centered")

    if vec.coloc.typeid != ColocationId.face_staggered:
        raise ValueError("Output vector field should be face staggered")

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
    # grad(u)_{x; i-1/2, j, k} = (u_{i, j, k} - u_{i-1, j, k})/dx
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                vx[i, j, k] = (u[i, j, k] - u[i-1, j, k])/(x[i] - x[i-1])
    
    #
    # grad(u)_{y; i, j-1/2, k} = (u_{i, j, k} - u_{i, j-1, k})/dy
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                vy[i, j, k] = (u[i, j, k] - u[i, j-1, k])/(y[j] - y[j-1])

    #
    # grad(u)_{z; i, j, k-1/2} = (u_{i, j, k} - u_{i, j, k-1})/dz
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                vz[i, j, k] = (u[i, j, k] - u[i, j, k-1])/(z[k] - z[k-1])



@numba.njit()
def divergence(vec, scalar):

    if vec.coloc.typeid != ColocationId.face_staggered:
        raise ValueError("Input vector field should be face staggered")

    if scalar.coloc.typeid != ColocationId.cell_centered:
        raise ValueError("Output scalar field should be cell centered")

    if not np.all(vec.mesh.num_cells == scalar.mesh.num_cells):
        raise ValueError("Fields with different number of cells given as inputs")
    
    
    div_u = scalar.data

    # Number of cells in total
    num_cells = vec.mesh.num_cells
    
    # Components of the input vector field: v_r, v_\theta, v_\phi 
    vr, vt, vp = vec.data

    # Mesh edge coordinates
    re, the, phe = vec.mesh.edges
    
    # Mesh center coordinates
    rc, thc, phc = vec.mesh.centers

    # Cell sizes
    dr, dth, dph = vec.mesh.spacing

    # Full range
    for i in range(0, num_cells[0]):
        for j in range(0, num_cells[1]):
            for k in range(0, num_cells[2]):

                rsin = rc[i]*np.sin(thc[j])

                div_u[i, j, k] \
                    =  (  (re[i+1]**2)*vr[i+1, j, k] \
                        - (re[i  ]**2)*vr[i  , j, k])/((rc[i]**2)*dr[i]) \
                     + (  np.sin(the[j+1])*vt[i, j+1, k] \
                        - np.sin(the[j  ])*vt[i, j  , k])/(rsin*dth[j]) \
                     + (vp[i, j, k+1] - vp[i, j, k])/(rsin*dph[k])
    

@numba.njit()
def curl_edge(invec, outvec):

    if invec.coloc.typeid != ColocationId.edge_staggered:
        raise ValueError("Input vector field should be edge staggered")

    if outvec.coloc.typeid != ColocationId.face_staggered:
        raise ValueError("Output vector field should be face staggered")

    if not np.all(invec.mesh.num_cells == outvec.mesh.num_cells):
        raise ValueError("Fields with different number of cells given as inputs")
            
    # Number of cells in total
    num_cells = invec.mesh.num_cells
    
    # Components of the input vector field
    vr, vt, vp = invec.data

    # Components of the output vector field
    curlv_r, curlv_t, curlv_p = outvec.data

    # Mesh edge coordinates
    re, te, pe = invec.mesh.edges
    
    # Mesh center coordinates
    rc, tc, pc = invec.mesh.centers

    # Cell sizes
    dr, dt, dp = invec.mesh.spacing

    #
    # curl(v)_r  @  (r_{i-1/2}, \theta_{j}, \phi_{k})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):

                rsint = re[i]*np.sin(tc[j])

                curlv_r[i, j, k] \
                    = (   (np.sin(te[j+1])*vp[i, j+1, k] - np.sin(te[j])*vp[i, j, k])/dt[j] \
                        - (                vt[i, j, k+1] -               vt[i, j, k])/dp[k] ) / rsint
                     
    #
    # curl(v)_\theta  @  (r_{i}, \theta_{j-1/2}, \phi_{k})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                curlv_t[i, j, k] \
                    = (   (        vr[i, j, k+1] -       vr[i, j, k])/(np.sin(te[j])*dp[k]) \
                        - (re[i+1]*vp[i+1, j, k] - re[i]*vp[i, j, k])/dr[i] ) / rc[i]
   
    #
    # curl(v)_\phi  @  (r_{i}, \theta_{j}, \phi_{k-1/2})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                curlv_p[i, j, k] \
                    = (  (re[i+1]*vt[i+1, j, k] - re[i]*vt[i, j, k])/dr[i] \
                       - (        vr[i, j+1, k] -       vr[i, j, k])/dt[j] )/rc[i]


@numba.njit()
def curl_face(invec, outvec):

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
    vr, vt, vp = invec.data

    # Components of the output vector field
    curlv_r, curlv_t, curlv_p = outvec.data

    # Mesh edge coordinates
    re, te, pe = invec.mesh.edges
    
    # Mesh center coordinates
    rc, tc, pc = invec.mesh.centers

    #
    # curl(v)_r  @  (r_{i}, \theta_{j-1/2}, \phi_{k-1/2})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                
                rsint = rc[i]*np.sin(te[j])

                curlv_r[i, j, k] \
                    = (  (  np.sin(tc[j  ])*vp[i, j  , k] 
                          - np.sin(tc[j-1])*vp[i, j-1, k])/(tc[j] - tc[j-1]) \
                       - (vt[i, j, k] - vt[i, j, k-1])/(pc[k]- pc[k-1]) )/(rsint)
                     
    #
    # curl(v)_\theta  @  (r_{i-1/2}, \theta_{j}, \phi_{k-1/2})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                
                curlv_t[i, j, k] \
                    = (  (vr[i, j, k] - vr[i, j, k-1])/((pc[k]-pc[k-1])*np.sin(tc[j])) \
                       - (  rc[i  ]*vp[i  , j, k] 
                          - rc[i-1]*vp[i-1, j, k])/(rc[i]-rc[i-1]) )/re[i]
   
    #
    # curl(v)_\phi  @  (r_{i-1/2}, \theta_{j-1/2}, \phi_{k})
    #
    for i in range(1, num_cells[0]):
        for j in range(1, num_cells[1]):
            for k in range(1, num_cells[2]):
                
                curlv_p[i, j, k] \
                    = (  (rc[i]*vt[i, j, k] - rc[i-1]*vt[i-1, j, k])/(rc[i]-rc[i-1]) \
                       - (vr[i, j, k] - vr[i, j-1, k])/(tc[j]-tc[j-1]) )/re[i]
   

@numba.njit()
def curl(invec, outvec):

    if invec.coloc.typeid == ColocationId.edge_staggered:
        curl_edge(invec, outvec)
    elif invec.coloc.typeid == ColocationId.face_staggered:
        curl_face(invec, outvec)
    else:
        raise ValueError("Input vector field should be either face or edge staggered")
