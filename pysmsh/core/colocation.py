# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Field data colocation definitions
"""


import numpy as np

import numba
import numba.experimental.jitclass as jitclass

import enum

class Colocation:
    
    typeid : numba.int32

    def __init__(self):
        self.typeid = 0
    
    def shapes(self, mesh, num):
        return [mesh.num_cells + self.add_to_shape(n) for n in range(num)]


class ColocationId(enum.IntEnum):
    cell_centered = 1
    node_centered = 2
    face_staggered = 3
    edge_staggered = 4


@jitclass()
class CellCentered(Colocation):

    def __init__(self):
        self.typeid = ColocationId.cell_centered

    def add_to_shape(self, n):
        return np.array((0, 0, 0))


@jitclass()
class NodeCentered(Colocation):
    
    def __init__(self):
        self.typeid = ColocationId.node_centered

    def add_to_shape(self, n):
        return np.array((1, 1, 1))


@jitclass()
class FaceStaggered(Colocation):
    
    def __init__(self):
        self.typeid = ColocationId.face_staggered

    def add_to_shape(self, n):
        return (np.array((1, 0, 0)), np.array((0, 1, 0)), np.array((0, 0, 1)))[n]


@jitclass()
class EdgeStaggered(Colocation):

    def __init__(self):
        self.typeid = ColocationId.edge_staggered

    def add_to_shape(self, n):
        return (np.array((0, 1, 1)), np.array((1, 0, 1)), np.array((1, 1, 0)))[n]


# Synonyms
Nodal = NodeCentered
Zonal = CellCentered


def get_colocation_from_string(s):

    colocmap \
        = {"nodal" : NodeCentered, 
           "zonal" : CellCentered,
           "cell_centered" : CellCentered,
           "node_centered" : NodeCentered,
           "face_staggered" : FaceStaggered,
           "edge_staggered" : EdgeStaggered
        }

    return colocmap[s.lower()]