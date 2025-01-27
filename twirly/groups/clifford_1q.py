# This code is part of Twirly.
#
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
Clifford twirling group for a single qubit
"""

from qiskit.quantum_info import Clifford

from ..utils import cached_property_by_dim
from .clifford import SmallCliffordGroup


class Clifford1Q(SmallCliffordGroup):
    r"""The one-qubit Clifford group."""

    def __init__(self):
        super().__init__(1)
        self.bootstrap()

    @cached_property_by_dim
    def _parent_orbits(self):
        x, z, h, s = map(Clifford.from_label, "XZHS")
        ofc = self._parent.orbit_from_clifford
        cosets = self._parent.outer(ofc(h @ s), ofc(h), ravel=True)
        paulis = self._parent.outer(ofc(x), ofc(z), ravel=True)
        return (cosets, paulis)

    def permute(self, members, permutations):  # pylint: disable=unused-argument
        # there is only one qubit so that no non-trivial permutations are possible
        return members.copy()
