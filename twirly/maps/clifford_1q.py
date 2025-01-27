# This code is part of Twirly.
#
# This is proprietary IBM software for internal use only, do not distribute outside of IBM
# Unauthorized copying of this file is strictly prohibited.
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
Single-qubit Clifford group maps
"""

from ..base import CachedMapMixin, TensoredInjection, TensoredInjectionComposition
from ..exceptions import TwirlingError
from ..groups import Clifford1Q, Clifford2Q, HaarUnitary, UniformClifford
from ..utils import shape_tuple
from .clifford import CliffordIntoUnitary, SmallCliffordGroupIntoUniformClifford


class Clifford1QIntoHaarUnitary(CachedMapMixin, TensoredInjectionComposition):
    def __init__(self):
        c1 = UniformClifford(1)
        map1 = CliffordIntoUnitary(c1, HaarUnitary(1))
        map2 = SmallCliffordGroupIntoUniformClifford(Clifford1Q(), c1)
        super().__init__(map1, map2)


class Clifford1QIntoClifford2Q(TensoredInjection):
    def __init__(self):
        super().__init__(Clifford1Q(), Clifford2Q())

    def apply(self, members):
        if members.ndim == 0 or members.shape[0] % 2 != 0:
            raise TwirlingError("The first dimension must exist and be even.")
        # Clifford2Q has .bounds (2, 10, 24, 24), with the last two indices corresponding to
        # single-qubit Cliffords
        return members[::2, ...].astype(self.codomain.dtype) + 24 * members[1::2, ...]

    def apply_reverse(self, members):
        members = self.codomain.unpack(members)
        if any(members[..., :2].ravel() > 0):
            raise TwirlingError("Cannot perform the reverse map on members outside of the image.")
        members = members[..., [3, 2]].astype(self.domain.dtype)
        members = members.transpose(shape_tuple(0, members.ndim - 1, range(1, members.ndim - 1)))
        return members.reshape((-1,) + members.shape[2:])

    def codomain_shapes(self, input_shape):
        return (shape_tuple(input_shape[0] // 2, input_shape[1:]),)

    def domain_shapes(self, output_shape):
        return (shape_tuple(output_shape[0] * 2, output_shape[1:]),)
