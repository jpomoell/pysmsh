# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Interpolation of staggered field data
"""

import numpy as np
import numba
import numba.experimental.jitclass as jitclass

from pysmsh.core.colocation import ColocationId


@jitclass()
class ReconstructionMoments:
    
    constant : numba.float64
    x : numba.float64;  y: numba.float64; z: numba.float64;
    xx : numba.float64;  yy: numba.float64; zz: numba.float64;
    xy : numba.float64;  xz: numba.float64; yz: numba.float64;
        
    def __init__(self):
        self.constant = 0.0
        self.x = self.y = self.z = 0.0
        self.xx = self.yy = self.zz = 0.0
        self.xy = self.xz = self.yz = 0.0


@jitclass()
class FaceStaggeredInterpolator:
        
    a : ReconstructionMoments
    b : ReconstructionMoments
    c : ReconstructionMoments
    
    xi: numba.float64;  yj: numba.float64;  zk: numba.float64;
    
    # Mesh centers
    centers: numba.types.UniTuple(numba.types.Array(numba.float64, 1, "C"), 3)
    
    # Mesh edges
    edges: numba.types.UniTuple(numba.types.Array(numba.float64, 1, "C"), 3)
        
    # Mesh spacing
    spacing: numba.types.UniTuple(numba.types.Array(numba.float64, 1, "C"), 3)

    # Domain extent
    indomain_extent: numba.types.UniTuple(numba.types.Array(numba.float64, 1, "C"), 3)

    # Reference to field data
    data : numba.types.UniTuple(numba.types.Array(numba.float64, 3, "C"), 3)
    
    div_free_interpolation : bool

    limit : bool
    
    def __init__(self, field, div_free_interpolation=True, limit=False):
        
        self.a = ReconstructionMoments()
        self.b = ReconstructionMoments()
        self.c = ReconstructionMoments()
        
        if field.coloc.typeid != ColocationId.face_staggered:
            raise ValueError("Input vector field should be face staggered")

        # Reference to data
        self.data = field.data
        
        # Copy grid coordinate information
        # Mainly done for performance
        self.centers = (field.mesh.centers[0], field.mesh.centers[1], field.mesh.centers[2])
        self.edges = (field.mesh.edges[0], field.mesh.edges[1], field.mesh.edges[2])
        self.spacing = (field.mesh.spacing[0], field.mesh.spacing[1], field.mesh.spacing[2])
        self.indomain_extent = (field.mesh.indomain_extent[0], field.mesh.indomain_extent[1], field.mesh.indomain_extent[2])
                
        self.div_free_interpolation = div_free_interpolation
        self.limit = limit
            
    def P1(self, x):
        return x
    
    def P2(self, x):
        return x*x
    
    def component(self, m, x, y, z):

        P1 = self.P1
        P2 = self.P2
        
        xi, yj, zk = self.xi, self.yj, self.zk
        
        first_order \
            = m.constant + m.x*P1(x - xi) + m.xy*P1(y - yj) + m.xz*P1(z - zk)

        second_order \
            = m.xy*P1(x - xi)*P1(y - yj) \
            + m.xz*P1(x - xi)*P1(z - zk) \
            + m.yz*P1(y - yj)*P1(z - zk) \
            + m.xx*P2(x - xi) \
            + m.yy*P2(y - yj) \
            + m.zz*P2(z - zk)
        
        return first_order + second_order

    def value(self, p):
        
        # Compute moments
        self.compute_moments(p)
        
        # Compute components
        x, y, z = p
        
        vx = self.component(self.a, x, y, z)
        vy = self.component(self.b, x, y, z)
        vz = self.component(self.c, x, y, z)
        
        return np.array((vx, vy, vz))
    
    def cartesian_unit_vector(self, p):
        
        v = self.value(p)
        
        vsqr = v[0]**2 + v[1]**2 + v[2]**2

        if vsqr > 0.0:
            return v/np.sqrt(vsqr)
        else:
            return np.zeros(3)
    
    def cell_size(self, p):

        i, j, k = self.index(p, self.edges)
        
        return np.array((self.spacing[0][i], self.spacing[1][j], self.spacing[2][k]))

    def index(self, p, crds):

        i = np.searchsorted(crds[0], p[0]) - 1 
        j = np.searchsorted(crds[1], p[1]) - 1
        k = np.searchsorted(crds[2], p[2]) - 1

        return i, j, k    
    
    def compute_moments(self, p):
                
        vx, vy, vz = self.data

        # Grid cell centers
        xc, yc, zc = self.centers
                
        # Cell index
        i, j, k = self.index(p, self.edges)
        
        # Set the cell center of the current cell
        self.xi, self.yj, self.zk = (xc[i], yc[j], zc[k])

        # Mesh spacing of current cell
        dx, dy, dz = self.spacing[0][i], self.spacing[1][j], self.spacing[2][k]
        
        # Compute derivatives
        # NOTE: these are second-order accurate only for uniform grids

        if self.limit:
            dvxdy_P = self.limiter(vx[i+1, j+1, k] - vx[i+1, j, k],
                                   vx[i+1, j, k] - vx[i+1, j-1, k])/(yc[j+1] - yc[j])

            dvxdy_M = self.limiter(vx[i, j+1, k] - vx[i, j, k],
                                   vx[i, j, k] - vx[i, j-1, k])/(yc[j+1] - yc[j])

            dvxdz_P = self.limiter(vx[i+1, j, k+1] - vx[i+1, j, k],
                                   vx[i+1, j, k] - vx[i+1, j, k-1])/(zc[k+1] - zc[k])

            dvxdz_M = self.limiter(vx[i, j, k+1] - vx[i, j, k],
                                   vx[i, j, k] - vx[i, j, k-1])/(zc[k+1] - zc[k])


            dvydx_P = self.limiter(vy[i+1, j+1, k] - vy[i, j+1, k],
                                   vy[i, j+1, k] - vy[i-1, j+1, k])/(xc[i+1] - xc[i])

            dvydx_M = self.limiter(vy[i+1, j, k] - vy[i, j, k],
                                   vy[i, j, k] - vy[i-1, j, k])/(xc[i+1] - xc[i])

            dvydz_P = self.limiter(vy[i, j+1, k+1] - vy[i, j+1, k],
                                   vy[i, j+1, k] - vy[i, j+1, k-1])/(zc[k+1] - zc[k])

            dvydz_M = self.limiter(vy[i, j, k+1] - vy[i, j, k],
                                   vy[i, j, k] - vy[i, j, k-1])/(zc[k+1] - zc[k])


            dvzdx_P = self.limiter(vz[i+1, j, k+1] - vz[i, j, k+1],
                                   vz[i, j, k+1] - vz[i-1, j, k+1])/(xc[i+1] - xc[i])

            dvzdx_M = self.limiter(vz[i+1, j, k] - vz[i, j, k],
                                   vz[i, j, k] - vz[i-1, j, k])/(xc[i+1] - xc[i])

            dvzdy_P = self.limiter(vz[i, j+1, k+1] - vz[i, j, k+1],
                                   vz[i, j, k+1] - vz[i, j-1, k+1])/(yc[j+1] - yc[j])

            dvzdy_M = self.limiter(vz[i, j+1, k] - vz[i, j, k],
                                   vz[i, j, k] - vz[i, j-1, k])/(yc[j+1] - yc[j])
        else:
            # dvx/dy  @  (x_{i+1/2}, y_j, z_k)
            dvxdy_P = (vx[i+1, j+1, k] - vx[i+1, j, k])/(yc[j+1] - yc[j])
            
            # dvx/dy  @  (x_{i-1/2}, y_j, z_k)
            dvxdy_M = (vx[i, j+1, k] - vx[i, j, k])/(yc[j+1] - yc[j])
            
            # dvx/dz  @  (x_{i+1/2}, y_j, z_k)
            dvxdz_P = (vx[i+1, j, k+1] - vx[i+1, j, k])/(zc[k+1] - zc[k])
            
            # dvx/dz  @  (x_{i-1/2}, y_j, z_k)
            dvxdz_M = (vx[i, j, k+1] - vx[i, j, k])/(zc[k+1] - zc[k])
            
            
            # dvy/dx  @  (x_i, y_{j+1/2}, z_k)
            dvydx_P = (vy[i+1, j+1, k] - vy[i, j+1, k])/(xc[i+1] - xc[i])
            
            # dvy/dx  @  (x_i, y_{j-1/2}, z_k)
            dvydx_M = (vy[i+1, j, k] - vy[i, j, k])/(xc[i+1] - xc[i])
            
            # dvy/dz  @  (x_i, y_{j+1/2}, z_k)
            dvydz_P = (vy[i, j+1, k+1] - vy[i, j+1, k])/(zc[k+1] - zc[k])
            
            # dvy/dz  @  (x_i, y_{j-1/2}, z_k)
            dvydz_M = (vy[i, j, k+1] - vy[i, j, k])/(zc[k+1] - zc[k])
            
            
            # dvy/dx  @  (x_i, y_j, z_{k+1/2})
            dvzdx_P = (vz[i+1, j, k+1] - vz[i, j, k+1])/(xc[i+1] - xc[i])
            
            # dvy/dx  @  (x_i, y_j, z_{k-1/2})
            dvzdx_M = (vz[i+1, j, k] - vz[i, j, k])/(xc[i+1] - xc[i])
            
            # dvz/dy  @  (x_i, y_j, z_{k+1/2})
            dvzdy_P = (vz[i, j+1, k+1] - vz[i, j, k+1])/(yc[j+1] - yc[j])
            
            # dvz/dy  @  (x_i, y_j, z_{k-1/2})
            dvzdy_M = (vz[i, j+1, k] - vz[i, j, k])/(yc[j+1] - yc[j])
       
        
        # Moments of the x-component of the field
        self.a.constant = (vx[i+1, j, k] + vx[i, j, k])/2.0
        
        self.a.x = (vx[i+1, j, k] - vx[i, j, k])/dx
        self.a.y = (dvxdy_M + dvxdy_P)/2.0
        self.a.z = (dvxdz_M + dvxdz_P)/2.0
        
        self.a.xy = (dvxdy_P - dvxdy_M)/dx
        self.a.xz = (dvxdz_P - dvxdz_M)/dx
        self.a.yz = 0.0
        
        self.a.xx = 0.0 # For non-div-free cases
        self.a.yy = 0.0
        self.a.zz = 0.0
        

        # Moments of the y-component of the field
        self.b.constant = (vy[i, j+1, k] + vy[i, j, k])/2.0
        
        self.b.x = (dvydx_M + dvydx_P)/2.0
        self.b.y = (vy[i, j+1, k] - vy[i, j, k])/dy
        self.b.z = (dvydz_M + dvydz_P)/2.0
        
        self.b.xy = (dvydx_P - dvydx_M)/dy
        self.b.xz = 0.0 
        self.b.yz = (dvydz_P - dvydz_M)/dy
        
        self.b.xx = 0.0 
        self.b.yy = 0.0 # For non-div-free cases
        self.b.zz = 0.0
        
        
        # Moments of the z-component of the field
        self.c.constant = (vz[i, j, k+1] + vz[i, j, k])/2.0
        
        self.c.x = (dvzdx_M + dvzdx_P)/2.0
        self.c.y = (dvzdy_M + dvzdy_P)/2.0
        self.c.z = (vz[i, j, k+1] - vz[i, j, k])/dz
        
        self.c.xy = 0.0
        self.c.xz = (dvzdx_P - dvzdx_M)/dz
        self.c.yz = (dvzdy_P - dvzdy_M)/dz
        
        self.c.xx = 0.0 
        self.c.yy = 0.0
        self.c.zz = 0.0 # For non-div-free cases
       
    
        if self.div_free_interpolation:
            
            self.a.xx = -(self.b.xy + self.c.xz)/2.0
            self.b.yy = -(self.a.xy + self.c.yz)/2.0
            self.c.zz = -(self.a.xz + self.b.yz)/2.0
        
            self.a.constant -= self.a.xx*dx*dx/4.0
            self.b.constant -= self.b.yy*dy*dy/4.0
            self.c.constant -= self.c.zz*dz*dz/4.0

    def limiter(self, dp, dm):

        limited_slope = 0.0

        if (dp*dm) > 0.0:
            limited_slope = 2.0*dp*dm/(dp+dm)

        return limited_slope
