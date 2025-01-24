# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Pauli twirling group
"""

import numpy as np

from ..base import NdEnumeratedTwirlingGroup


class UniformPauli(NdEnumeratedTwirlingGroup):
    def __init__(self):
        super().__init__(1)

    @property
    def dtype(self):
        return np.int8

    @property
    def bounds(self):
        # boolean powers (i, j) of the X^i @ Z^j
        return (2, 2)

    def dot(self, lhs, rhs):
        return np.bitwise_xor(lhs, rhs)

    def inv(self, members):
        # paulis are self-inverse
        return members

    def permute(self, members, permutations):
        # there is only one qubit so that no non-trivial permutations are possible
        return members.copy()
