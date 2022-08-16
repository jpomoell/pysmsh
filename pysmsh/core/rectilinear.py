# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Rectilinear mesh class
"""

import numpy as np

import pysmsh.core.nutils as nutils



class RectilinearMeshBase:
     
    num_ghost_cells: int

    def __init__(self, axes, num_ghost_cells=0):
        self.axes = axes
        self.num_ghost_cells = num_ghost_cells
    
    def _coordinates(self, crds):
        return nutils.create_tuple_like_from_seq_copy(self._CoordinateContainer, crds)
    
    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the mesh
        """
        return len(self.axes)

    @property
    def num_cells(self) -> np.ndarray:
        """Returns the number of cells in the mesh
        """
        return np.array([axis.num-1 for axis in self.axes])

    @property
    def centers(self):
        """Returns cell center coordinates of the mesh
        """
        return self._coordinates([axis.centers for axis in self.axes])
        
    @property
    def spacing(self):
        """Returns mesh cell sizes
        """
        return self._coordinates([axis.spacing for axis in self.axes])
   
    @property
    def edges(self):
        """Returns cell edge coordinates of the mesh
        """
        return self._coordinates([axis.edges for axis in self.axes])

    @property
    def extent(self):
        """Returns the extent of the entire domain
        """
        return self._coordinates([axis.extent for axis in self.axes])

    @property
    def indomain_extent(self):
        """Returns the extent of the in-domain region
        """
        edges = self.indomain_edges
        
        return self._coordinates([np.array((np.min(edge), np.max(edge))) for edge in edges])

    @property
    def indomain_edges(self):
        """Returns cell edge coordinates of in-domain region
        """
        beg_idx = self.num_ghost_cells
        end_idx = self.num_cells - self.num_ghost_cells + 1

        return self._coordinates([self.axes[d].edges[beg_idx:end_idx[d]] for d in range(self.ndim)])
        
    def face_center_coords(self, dim):
        """Returns the face-centered coordinates for the faces with normals
        in the dim:th coordinate direction
        """
        crds = [self.axes[i].centers for i in range(self.ndim)]
        crds[dim] = self.axes[dim].edges

        return self._coordinates(crds)

    def edge_center_coords(self, dim):
        """Returns the edge-centered coordinates for the cell edges directed
        in the dim:th coordinate direction
        """
        crds = [self.axes[i].edges for i in range(self.ndim)]
        crds[dim] = self.axes[dim].centers

        return self._coordinates(crds)