# This file is part of pySMSH.
#
# Copyright 2022 pySMSH developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Partition class
"""

import numpy as np
import numba
import numba.types
import numba.experimental.jitclass as jitclass


@jitclass()
class DiscreteInterval:
    """Class defining a partition, i.e. an interval constructed as a union of subintervals. 
    
    The discrete interval models the concept of a partition, an ordered set of strictly 
    increasing numbers. The given numbers constitute end and starting points of subintervals, 
    such that the entire interval is a union of the subintervals.
    
    Attributes:
        name (str)         : string describing the interval
        elements (ndarray) : the numbers that define the subintervals
    """
  
    name: str
    elements: numba.types.Array(numba.float64, 1, "C")

    def __init__(self, name, elements):
        self.name = name
        self.elements = np.asarray(elements)

    @property
    def num(self) -> int:
        """Returns the number of elements in the interval.
        """
        return len(self.elements)
    
    @property
    def spacing(self) -> np.ndarray:
        """Returns the distance between the elements in the interval
        """
        return self.elements[1::] - self.elements[0:-1]

    @property
    def edges(self) -> np.ndarray:
        """Returns the values of the elements
        """
        return self.elements

    @property
    def centers(self) -> np.ndarray:
        """Returns the averages of the element values
        """
        return 0.5*(self.elements[0:-1] + self.elements[1::])

    @property
    def extent(self) -> np.ndarray:
        """Returns the extent of the interval
        """
        return np.array((self.elements[0], self.elements[-1]))


def extend_interval(interval, num):
    """Extends a discrete interval by adding coordinates at each end.

    Adds given number of coordinates at each end of the interval.
    The addition is made using the same coordinate spacing as found
    at the ends.

    Args:
        interval : The input interval to modify
        num :  number of coordinates to insert at each end
    Returns:
        A new interval that has been extended
    """

    if num == 0:
        return DiscreteInterval(interval.name, np.copy(interval.elements))

    x = interval.elements
    
    new_elems = np.zeros(len(x) + 2*num)
    new_elems[num:-num] = x[:]

    for idx in range(0, num):
        
        # lower edge
        new_elems[idx] = x[0]  - (num-idx)*(x[1]-x[0])
        
        # upper edge:
        new_elems[idx+len(x)+num]  = x[-1] + (idx+1)*(x[-1]-x[-2])

    return DiscreteInterval(interval.name, new_elems)


