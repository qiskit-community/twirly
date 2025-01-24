# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Enumerated twirling group base classes
"""

import abc

import numpy as np

from ..utils import cached_property_by_dim, get_rng, shape_tuple
from .twirling_group import TwirlingGroup

ENUMERATED_DTYPE = np.int16


class EnumeratedTwirlingGroup(TwirlingGroup):
    r"""
    Base class for finitely enumerated twirling groups.

    Each group member is represented as a non-negative integer less than :attr:`~num_members`. In
    particular, there can be no gaps in the enumeration. Note that the :attr:`~shape` is always the
    empty tuple ``()`` because no extra dimensions are needed to describe group members.
    """

    @property
    def all_members(self):
        r"""
        A 1D member array containing all group members, in order.
        """
        return np.arange(self.num_members, dtype=self.dtype)

    @property
    def dtype(self):
        return ENUMERATED_DTYPE

    @property
    @abc.abstractclassmethod
    def num_members(self):
        r"""
        The order of the group; the total number of group members.
        """
        pass

    @property
    def shape(self):
        return ()

    def equal(self, lhs, rhs):
        # override np.allclose for performance reasons
        return lhs.shape == rhs.shape and all((lhs == rhs).ravel())

    def id_member(self, shape=()):
        return np.zeros(shape, dtype=self.dtype)

    def sample(self, shape, seed=None):
        return get_rng(seed).integers(0, self.num_members, size=shape, dtype=self.dtype)


class NdEnumeratedTwirlingGroup(EnumeratedTwirlingGroup):
    r"""
    A base class for the special case of :class:`~EnumeratedTwirlingGroup` where the members are
    structured into a n-dimensional lexicographical ordering. For example, the single-qubit
    Cliffords are often broken into six cosets of the Pauli subgroup, in which case a child class
    might implement :attr:`~bounds` to return ``(6,4)``
    """

    @cached_property_by_dim
    def _idx_converter(self):
        prod = [1]
        for bound in reversed(self.bounds[1:]):
            prod.append(prod[-1] * bound)
        return np.array(prod[::-1], dtype=self.dtype)

    @property
    @abc.abstractmethod
    def bounds(self):
        pass

    @property
    def num_members(self):
        return int(np.prod(self.bounds, dtype=int))

    def unpack(self, members, copy=True):
        if copy:
            members = members.copy()
        ret = np.empty(shape_tuple(members.shape, len(self.bounds)), dtype=self.dtype)
        idx = ret.shape[-1]
        for bound in reversed(self.bounds):
            idx -= 1
            ret[..., idx] = members % bound
            if idx:
                members //= bound
        return ret

    def pack(self, unpacked_members):
        return unpacked_members @ self._idx_converter
