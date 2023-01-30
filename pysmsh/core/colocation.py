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
    face0_centered = 5
    face1_centered = 6
    face2_centered = 7
    edge0_centered = 8
    edge1_centered = 9
    edge2_centered = 10


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


@jitclass()
class Face0Centered(Colocation):

    def __init__(self):
        self.typeid = ColocationId.face0_centered

    def add_to_shape(self, n):
        return np.array((1, 0, 0))


@jitclass()
class Face1Centered(Colocation):

    def __init__(self):
        self.typeid = ColocationId.face1_centered

    def add_to_shape(self, n):
        return np.array((0, 1, 0))


@jitclass()
class Face2Centered(Colocation):

    def __init__(self):
        self.typeid = ColocationId.face2_centered

    def add_to_shape(self, n):
        return np.array((0, 0, 1))


@jitclass()
class Edge0Centered(Colocation):

    def __init__(self):
        self.typeid = ColocationId.edge0_centered

    def add_to_shape(self, n):
        return np.array((0, 1, 1))


@jitclass()
class Edge1Centered(Colocation):

    def __init__(self):
        self.typeid = ColocationId.edge1_centered

    def add_to_shape(self, n):
        return np.array((1, 0, 1))


@jitclass()
class Edge2Centered(Colocation):

    def __init__(self):
        self.typeid = ColocationId.edge2_centered

    def add_to_shape(self, n):
        return np.array((1, 1, 0))


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
           "edge_staggered" : EdgeStaggered,
           "face0_centered" : Face0Centered,
           "face1_centered" : Face1Centered,
           "face2_centered" : Face2Centered,
           "edge0_centered" : Edge0Centered,
           "edge1_centered" : Edge1Centered,
           "edge2_centered" : Edge2Centered,
        }

    return colocmap[s.lower()]