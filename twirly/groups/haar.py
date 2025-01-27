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
Haar uniform twirling group
"""

import numpy as np
from qiskit.quantum_info import random_unitary

from ..base import TwirlingGroup
from ..exceptions import TwirlingError
from ..utils import get_rng, iter_as_shape, shape_tuple


class HaarUnitary(TwirlingGroup):
    @property
    def dtype(self):
        return np.complex128

    @property
    def shape(self):
        return (self.dim,) * 2

    def equal(self, lhs, rhs) -> bool:
        try:
            prod = self.ravel(self.dot(lhs, self.inv(rhs)))
        except ValueError:
            # not broadcastable
            return False
        overlaps = np.abs(prod.reshape(-1, self.dim**2)[:, :: self.dim + 1].sum(axis=1))
        return np.allclose(overlaps, self.dim)

    def id_member(self, shape=()):
        ret = np.zeros(shape_tuple(shape, self.shape), dtype=self.dtype)
        ret.reshape(-1, self.dim**2)[:, :: (self.dim + 1)] = 1
        return ret

    def inv(self, members):
        return members.transpose(shape_tuple(range(members.ndim - 2), -1, -2)).conjugate()

    def from_operation(self, operation):
        if operation.num_qubits != self.num_qubits:
            raise TwirlingError(
                "{self} can only convert operations with {num_qubits}={self.num_qubits}, but "
                "encountered {operation}."
            )
        try:
            member = operation.to_matrix()
        except Exception as exc:
            raise TwirlingError(
                f"There was a problem converting {operation} to a member of {self}."
            ) from exc
        return np.array(member, dtype=self.dtype, copy=False)

    def dot(self, lhs, rhs):
        return np.matmul(lhs, rhs)

    def permute(self, members, permutations):
        # TODO: this implementation is inefficient
        shape = np.broadcast_shapes(members.shape[:-2], permutations.shape[:-1])
        ret = np.empty(shape_tuple(shape, [2] * (2 * self.num_qubits)), dtype=self.dtype)
        ret.ravel()[...] = members.ravel()

        permutations = permutations[(None,) * (len(shape) - permutations.ndim + 1) + (...,)]
        for member, perm in zip(iter_as_shape(ret, shape), iter_as_shape(permutations, shape)):
            perm = np.concatenate([perm, perm + self.num_qubits])
            member[...] = member.transpose(perm)
        return ret.reshape(shape_tuple(shape, self.shape))

    def sample(self, shape=(), seed=None):
        rng = get_rng(seed)
        ret = self.empty(shape)
        for member in self.ravel(ret):
            member[:, :] = random_unitary(self.dim, seed=rng)
        return ret
