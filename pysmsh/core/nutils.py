# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Numba utilities
"""

import numpy as np
import numba


@numba.njit()
def create_tuple_like_from_seq_copy(tup, seq):
    """Creates a new instance of type tup initialized with
    data from a sequence
    """

    if len(tup) == 1:
        newtup = type(tup)(np.copy(seq[0]))
    elif len(tup) == 2:
        newtup = type(tup)(np.copy(seq[0]), np.copy(seq[1]))
    elif len(tup) == 3:
        newtup = type(tup)(np.copy(seq[0]), np.copy(seq[1]), np.copy(seq[2]))
    elif len(tup) == 4:
        newtup = type(tup)(np.copy(seq[0]), np.copy(seq[1]), np.copy(seq[2]), np.copy(seq[3]))
    else:
        raise NotImplementedError("Unsupported number of dimensions in given sequence")
        
    return newtup

