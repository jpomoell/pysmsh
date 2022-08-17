# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Streamline tracer
"""
import typing
import numpy as np
import numba
import numba.types
import numba.experimental.jitclass as jitclass


@jitclass()
class MidPointStreamLineTracer:
    """Stream line tracer.

    Traces stream lines given a vector field defined on a grid. The integration
    is performed using a second-order midpoint scheme.

    NOTE: Periodic directions not explicitly supported

    Attributes:
        follow_field_direction (bool)  : Follow field line in the direction
                                         of the field or opposite to it
        max_path_length (float)        : Maximum path length after which tracing is stopped
        min_step_size (float)          : Limit the minimum allowable step size
        step_size_relative_to_mesh_spacing (float)  : Length of step relative to the local grid size
    """

    max_path_length : numba.float64
    min_step_size : numba.float64
    relative_step_size : numba.float64
    follow_field_direction : bool
    record_points : bool

    integral : numba.float64

    domain_min : numba.types.Array(numba.float64, 1, "C")
    domain_max : numba.types.Array(numba.float64, 1, "C")
    _domain_extent_initialized : bool

    points : numba.types.List(numba.types.Array(numba.types.float64, 1, "C")) 
    
    def __init__(self, 
                 max_path_length, 
                 relative_step_size=1.0, 
                 follow_field_direction=True, 
                 min_step_size=-1.0, 
                 record_points=False):

        self.max_path_length = max_path_length

        self.relative_step_size = relative_step_size

        self.follow_field_direction = follow_field_direction

        self.record_points = record_points

        self.min_step_size = min_step_size
        if self.min_step_size < 0.0:
            self.min_step_size = 1e-4*max_path_length

        self._domain_extent_initialized = False
        self.domain_max = np.zeros(3)
        self.domain_min = np.zeros(3)

        # Hack to easily create an empty list of the desired type
        self.points = [np.zeros(3)]
        self.points.clear()

    def is_inside_domain(self, pt):
        """Checks if the given point is inside the domain.
        """
        x, y, z = pt

        is_inside = False
        if      (x > self.domain_min[0]) and (x < self.domain_max[0]) \
            and (y > self.domain_min[1]) and (y < self.domain_max[1]) \
            and (z > self.domain_min[2]) and (z < self.domain_max[2]):
            is_inside = True

        return is_inside

    def intersection(self, xout, xin):

        # Find intersection between ray
        #
        # x = v t + x0 = (xout - xin)t + xin  (t should be > 0)
        #
        # and plane: dot(n, x) - dot(n, p) = 0  (p point on plane, n normal)

        v = xout - xin
        
        # Possible values for t
        t = np.zeros(6)

        # Intersects xmin plane
        p = np.array((self.domain_min[0], 0.0, 0.0))
        n = np.array((-1.0, 0.0, 0.0))
        if np.dot(n, v) != 0.0:
            t[0] = (np.dot(n, p) - np.dot(n, xin))/np.dot(n, v)

        # Intersects xmax plane
        p = np.array((self.domain_max[0], 0.0, 0.0))
        n = np.array((1.0, 0.0, 0.0))
        if np.dot(n, v) != 0.0:
            t[1] = (np.dot(n, p) - np.dot(n, xin))/np.dot(n, v)

        # Intersects ymin plane
        p = np.array((0.0, self.domain_min[1], 0.0))
        n = np.array((0.0, -1.0, 0.0))
        if np.dot(n, v) != 0.0:
            t[2] = (np.dot(n, p) - np.dot(n, xin))/np.dot(n, v)

        # Intersects ymax plane
        p = np.array((0.0, self.domain_max[1], 0.0))
        n = np.array((0.0, 1.0, 0.0))
        if np.dot(n, v) != 0.0:
            t[3] = (np.dot(n, p) - np.dot(n, xin))/np.dot(n, v)

        # Intersects zmin plane
        p = np.array((0.0, 0.0, self.domain_min[2]))
        n = np.array((0.0, 0.0, -1.0))
        if np.dot(n, v) != 0.0:
            t[4] = (np.dot(n, p) - np.dot(n, xin))/np.dot(n, v)

        # Intersects zmax plane
        p = np.array((0.0, 0.0, self.domain_max[2]))
        n = np.array((0.0, 0.0, 1.0))
        if np.dot(n, v) != 0.0:
            t[5] = (np.dot(n, p) - np.dot(n, xin))/np.dot(n, v)

        t_pos = t[np.where(t > 0.0)]
        t_min = np.min(t_pos)
        
        return v*t_min + xin
        
    def compute(self, start_pt, interpolator, integrand=None):
        """Computes the stream line

        Args:
            start_point (array) : Coordinates of the starting point
            interpolator        : Interpolator class

        Returns:
            Coordinates of the end point of the field line
        """

        if not self._domain_extent_initialized:
            for d in (0, 1, 2):
                self.domain_max[d] = np.max(interpolator.indomain_extent[d])
                self.domain_min[d] = np.min(interpolator.indomain_extent[d])

            self._domain_extent_initialized = True

        # Starting point
        x = np.array(start_pt, dtype=numba.float64)

        # Break if point outside
        if not self.is_inside_domain(x):
            return x

        # Record point
        if self.record_points:
            self.points.clear()
            self.points.append(np.copy(x))

        # Value of path integral
        self.integral = 0.0

        # Scale each step by a constant
        direction = 1.0 if self.follow_field_direction else -1.0
        scale_step_size = self.relative_step_size*direction

        path_length = 0.0
        while path_length < self.max_path_length:
            
            # Get cell size at current position
            cell_size = interpolator.cell_size(x)
            
            # Determine step size
            ds = max(np.min(cell_size), self.min_step_size)
            ds *= scale_step_size

            # Move half step
            xhalf = x + 0.5*ds*interpolator.cartesian_unit_vector(x)

            # Halfway point outside the domain?
            if not self.is_inside_domain(xhalf):
                x = self.intersection(xhalf, x)
                break

            # Move full step using the field at the half step
            xnext = x + ds*interpolator.cartesian_unit_vector(xhalf)

            # Next point outside the domain?
            if not self.is_inside_domain(xnext):
                x = self.intersection(xnext, xhalf)
                break

            # Add segment to path length
            path_length += np.abs(ds)

            # Add contribution to path integral
            if integrand is not None:
                self.integral += np.abs(ds)*integrand.compute(xhalf)

            # Update point
            x = xnext

            # Record
            if self.record_points:
                self.points.append(np.copy(x))

        if self.record_points:
            self.points.append(np.copy(x))

        return x