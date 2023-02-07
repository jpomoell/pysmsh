# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Mesh creation routines
"""


import numpy as np
import typing

import multimethod

import numba
import numba.types
import numba.experimental.jitclass as jitclass

import pysmsh
import pysmsh.core
from pysmsh.core.dinterval import DiscreteInterval
from pysmsh.core.rectilinear import RectilinearMeshBase


def _create_coordinates_container(names):
    return numba.types.namedtuple("Coordinates", names)


class Mesh:
    
    @multimethod.multimethod
    def Rectilinear(axes, num_ghost_cells=0, includes_ghost_coords=False):

        for axis in axes:
            if not isinstance(axis, DiscreteInterval):
                raise ValueError(f"Cannot crate mesh from given input. Expected DiscreteInterval but got {type(axis)}")

        # Add coordinates for ghosts if not present
        if not includes_ghost_coords:
            for i in range(len(axes)):
                axes[i] = pysmsh.core.dinterval.extend_interval(axes[i], num_ghost_cells)

        # Axis names
        names = [axis.name for axis in axes]
        
        # Number of dimensions
        ndim = len(names)

        # Create coordinate container type
        Coordinates = _create_coordinates_container(names)
        
        cont = Coordinates(*[np.zeros(2) for d in range(ndim)])

        # Generate spec for the mesh type
        spec = [('_CoordinateContainer', numba.typeof(cont)),
                ('axes', numba.types.UniTuple(DiscreteInterval.class_type.instance_type, ndim))]

        # Returned generated mesh type
        RectilinearMesh = jitclass(spec)(RectilinearMeshBase)       

        # Note: Make sure input is a tuple
        return RectilinearMesh(tuple(axes), num_ghost_cells)

    @Rectilinear.register
    def from_dict(container : typing.Mapping[str, np.ndarray], *, num_ghost_cells=0, includes_ghost_coords=False):

        axes = list()

        for name, values in container.items():
            axes.append(DiscreteInterval(name, values))

        return Mesh.Rectilinear(axes, num_ghost_cells, includes_ghost_coords)

    @Rectilinear.register
    def from_list_of_tuples(container : list[tuple[str, np.ndarray]], *, num_ghost_cells=0, includes_ghost_coords=False):

        axes = list()

        for item in container:
            name, values = item
            axes.append(DiscreteInterval(name, values))

        return Mesh.Rectilinear(axes, num_ghost_cells, includes_ghost_coords)

