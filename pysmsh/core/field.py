# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Field data structures
"""

import numpy as np

import numba
import numba.types
import numba.experimental.jitclass as jitclass

import pysmsh.core.colocation


class FieldDataContainer:
    
    _num_components : int
    
    def __init__(self):
        self._num_components = 1

    @property
    def num_components(self):
        return self._num_components
    
    @property
    def is_vector_field(self):
        return False
    
    @property
    def is_scalar_field(self):
        return False


class ScalarField(FieldDataContainer):
        
    @property
    def is_scalar_field(self):
        return True


class VectorField(FieldDataContainer):
        
    @property
    def is_vector_field(self):
        return True


class FieldType:

    @staticmethod
    def Scalar(mesh, colocation):

        # Numba spec of the type
        spec = [("mesh", numba.typeof(mesh)),
                ("coloc", numba.typeof(colocation)),
                ("data", numba.types.Array(numba.float64, mesh.ndim, "C"))]
    
        # Return the type
        return jitclass(ScalarField, spec)

    @staticmethod
    def Vector(mesh, colocation, num_components):

        # Numba spec of the type
        spec = [("mesh", numba.typeof(mesh)),
                ("coloc", numba.typeof(colocation)),
                ("data", numba.types.UniTuple(numba.types.Array(numba.float64, mesh.ndim, "C"), num_components))]
    
        # Return the type
        return jitclass(VectorField, spec)


class Field:

    def Scalar(mesh, colocation):
    
        if isinstance(colocation, str):
            coloc = pysmsh.core.colocation.get_colocation_from_string(colocation)()
        else:
            coloc = colocation()

        scalar = FieldType.Scalar(mesh, coloc)()
        scalar.mesh = mesh
        scalar.coloc = coloc

        # Allocate & initialize the data to zeros
        shape = coloc.shapes(mesh, 1)[0]
        scalar.data = np.zeros(shape)
                
        return scalar

    def Vector(mesh, colocation, num_components=3):
        
        if isinstance(colocation, str):
            coloc = pysmsh.core.colocation.get_colocation_from_string(colocation)()
        else:
            coloc = colocation()

        vector = FieldType.Vector(mesh, coloc, num_components)()
        vector._num_components = num_components
        vector.mesh = mesh
        vector.coloc = coloc

        # Allocate & initialize the data to zeros
        shapes = coloc.shapes(mesh, vector.num_components)

        data = list()
        for i in range(vector.num_components):
            data.append(np.zeros(shapes[i]))

        vector.data = tuple(data)

        return vector