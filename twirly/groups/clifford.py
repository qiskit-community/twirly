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
Clifford twirling groups
"""

import abc
from functools import partial
from itertools import product
from typing import Callable

import numpy as np
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.random import random_clifford

from ..base import NdEnumeratedTwirlingGroup, TwirlingGroup
from ..exceptions import TwirlingError
from ..utils import cached_property_by_dim, get_rng, iter_as_shape, shape_tuple

_UnsafeClifford = partial(Clifford, validate=False)


# A lookup table for calculating phases. The indices are:
# current_x, current_z, running_x_count, running_z_count
_PHASE_LOOKUP = np.array([0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 0, -1, 0, -1, 1, 0]).reshape(2, 2, 2, 2)


def apply_clifford(clifford_tableau, tableau):
    r"""Compute the action of an :math:`n`\-Clifford tableau on any other tableau.

    Both inputs should have the block-column structure ``x | z | phases`` of respective widths
    :math:`n`, :math`:`n`, and :math:`1`. When the second tableau happens to represent a Clifford,
    this method implements Clifford multiplication.

    Args:
        clifford_tableau: The Clifford to apply.
        tableau: The tableau to apply to.

    Returns:
        The action of the Clifford on the tableau.
    """
    num_qubits = clifford_tableau.shape[0] // 2

    # alias some slices
    c_x = clifford_tableau[:, :num_qubits].astype(np.uint8)
    c_z = clifford_tableau[:, num_qubits : 2 * num_qubits].astype(np.uint8)
    c_paulis = clifford_tableau[:, : 2 * num_qubits]
    c_phases = clifford_tableau[:, -1]

    s_paulis = tableau[:, : 2 * num_qubits]
    s_phases = tableau[:, -1]

    # Correcting for phase due to Pauli multiplication. Start with factors of -i from XZ = -iY
    # on individual qubits, and then handle multiplication between each qubitwise pair.
    i_factors = np.sum(s_paulis[:, :num_qubits] & s_paulis[:, num_qubits:], axis=1, dtype=int)

    for idx, s_pauli in enumerate(s_paulis):
        c_x_accum = np.logical_xor.accumulate(c_x_select := c_x[s_pauli], axis=0).astype(np.uint8)
        c_z_accum = np.logical_xor.accumulate(c_z_select := c_z[s_pauli], axis=0).astype(np.uint8)
        indexer = (c_x_select[1:], c_z_select[1:], c_x_accum[:-1], c_z_accum[:-1])
        i_factors[idx] += _PHASE_LOOKUP[indexer].sum()
    phases = np.mod(i_factors, 4) // 2

    # construct output tableau
    ret = np.empty((tableau.shape[0], 2 * num_qubits + 1), dtype=bool)
    ret[:, -1] = (np.matmul(s_paulis, c_phases, dtype=int) + s_phases + phases) % 2
    ret[:, :-1] = np.matmul(s_paulis, c_paulis, dtype=int) % 2

    return ret


class UniformClifford(TwirlingGroup):
    r"""
    The `n`-qubit Clifford grup.
    """

    @property
    def dtype(self):
        return bool

    @property
    def shape(self):
        return (2 * self.num_qubits, 2 * self.num_qubits + 1)

    @staticmethod
    def _hash(members):
        packed = np.packbits(members.reshape(members.shape[:-2] + (-1,)), axis=-1)
        return np.apply_along_axis(lambda val: hash(val.tobytes()), -1, packed)

    def id_member(self, shape=()):
        # flattening last two dimensions provides vectorized way to make identity matrices
        d = 2 * self.num_qubits
        ret = np.zeros(shape_tuple(shape, d * (d + 1)), dtype=self.dtype)
        ret[..., :: (self.dim + 2)] = True
        return ret.reshape(shape_tuple(shape, self.shape))

    def equal(self, lhs, rhs):
        # override np.allclose for performance reasons
        return lhs.shape == rhs.shape and all((lhs == rhs).ravel())

    def inv(self, members):
        ret = np.empty_like(members)
        flat_ret = self.ravel(ret)
        for idx, cliff in enumerate(self.to_cliffords(members)):
            flat_ret[idx, ...] = cliff.adjoint().tableau
        return ret

    def dot(self, lhs, rhs):
        # this method works when the RHS stores arbitrary tableaus, not just clifford tableaus
        l_shape = lhs.shape[-2:]
        r_shape = rhs.shape[-2:]
        shape = np.broadcast_shapes(lhs.shape[:-2], rhs.shape[:-2])
        ret = np.empty(shape_tuple(shape, r_shape), dtype=self.dtype)

        lhs = np.broadcast_to(lhs, shape_tuple(shape, l_shape)).reshape(-1, *l_shape)
        rhs = np.broadcast_to(rhs, shape_tuple(shape, r_shape)).reshape(-1, *r_shape)

        flat_ret = ret.reshape(-1, *ret.shape[-2:])
        for idx, (c1, c2) in enumerate(zip(lhs, rhs)):
            flat_ret[idx, ...] = apply_clifford(c1, c2)
        return ret

    def from_operation(self, operation):
        try:
            clifford = Clifford(operation)
        except Exception as exc:
            raise TwirlingError("Could not parse on or more operations as a Clifford.") from exc
        if clifford.num_qubits != self.num_qubits:
            raise TwirlingError("The provided Clifford operation has the wrong size.")
        return np.array(clifford.tableau, dtype=self.dtype)

    def orbit_from_clifford(self, clifford):
        return self.find_orbit(self.from_operation(clifford))

    def permute(self, members, permutations):
        # TODO: this implementation is inefficient
        assert members.ndim == permutations.ndim + 1
        shape = np.broadcast_shapes(members.shape[:-2], permutations.shape[:-1])

        ret = self.empty(shape)
        ret[...] = members
        for member, perm in zip(self.ravel(ret), iter_as_shape(permutations, shape)):
            perm = np.concatenate([perm, perm + self.num_qubits])
            member[...] = member[..., perm, :]
            member[..., :, :-1] = member[..., perm]
        return ret

    def propagate_paulis(self, members, paulis, phases=np.array([0], dtype=bool)):
        """Apply the Clifford operations to Paulis, including phase propagation.

        This function uses a uint8 representation for Paulis where I=0, Z=1, X=2, Y=iZX=3.

        The last two dimensions of ``paulis`` specify the symplectic portion of a tableau, such that
        the last dimension represents an n-qubit Pauli (in the representation described above), and
        where the second last dimension is over rows.
        The ``phases` argument, when broadcasted, gives a (boolean) phase to each Pauli, and
        therefore its last index should be over rows.

        Args:
            members: An array of Clifford tableaus.
            paulis: An array of Paulis, as described above.
            phases: An array of phases to associate to the Paulis. The last axis must be
                broadcastable against the second last axis of ``paulis``.

        Returns:
            A pair ``paulis, phases``
        """
        tableau_shape = np.broadcast_shapes(paulis.shape[:-2], phases.shape[:-1])
        num_qubits = members.shape[-2] // 2
        num_rows = np.broadcast_shapes(paulis.shape[-2], phases.shape[-1])

        tableau = np.empty((*tableau_shape, *num_rows, 2 * num_qubits + 1), dtype=self.dtype)
        tableau[..., :num_qubits] = np.bitwise_and(paulis, 2)
        tableau[..., num_qubits : 2 * num_qubits] = np.bitwise_and(paulis, 1)
        tableau[..., -1] = phases

        tableau = self.dot(members, tableau)
        paulis = np.left_shift(tableau[..., :num_qubits].astype(np.uint8), 1)
        paulis |= tableau[..., num_qubits : 2 * num_qubits].astype(np.uint8)
        return paulis, tableau[..., -1]

    def sample(self, shape=(), seed=None):
        rng = get_rng(seed)
        ret = self.empty(shape)
        flat_ret = self.ravel(ret)
        for idx in range(flat_ret.shape[0]):
            flat_ret[idx, :, :] = random_clifford(self.num_qubits, rng).tableau
        return ret

    def to_cliffords(self, members):
        yield from map(_UnsafeClifford, self.ravel(members))


class SmallCliffordGroup(NdEnumeratedTwirlingGroup):
    def __init__(self, num_qubits):
        super().__init__(num_qubits)
        # inheriting from UniformClifford is more trouble than its worth because the member shape
        # is different, so that many more things than expected would need to be overloaded
        self._parent = UniformClifford(num_qubits)

    def _lookup_parent_members(self, parent_members):
        return self._parent_lookup(self._parent._hash(parent_members))

    @cached_property_by_dim
    def _parent_lookup(self) -> Callable:
        lookup = {h: idx for idx, h in enumerate(UniformClifford._hash(self._all_parent_members))}
        if len(lookup) != self.num_members:
            raise TwirlingError(f"Duplicate Cliffords or hash collision detected in {type(self)}.")
        return np.vectorize(lookup.__getitem__)

    @cached_property_by_dim
    def _all_parent_members(self):
        return self._parent.outer(*self._parent_orbits, ravel=True)

    def _full_dot(self, lhs, rhs):
        return self._full_dot_table[lhs, rhs]

    @cached_property_by_dim
    def _full_dot_table(self):
        all_members = self.all_members
        return self.outer(all_members, all_members)

    def _full_inv(self, members):
        return self._full_inv_table[members]

    @cached_property_by_dim
    def _full_inv_table(self):
        return self.inv(self.all_members)

    def _full_propagate_paulis(self, members, paulis, phases=np.array([0])):
        paulis_lookup, phases_lookup, indexer = self._full_propagate_paulis_table
        idxs = paulis.astype(int) @ indexer
        members = members[..., None]
        return paulis_lookup[members, idxs], phases_lookup[members, idxs] ^ phases

    @cached_property_by_dim
    def _full_propagate_paulis_table(self):
        all_paulis = np.array(
            list(map(list, product([0, 1, 2, 3], repeat=self.num_qubits))),
            dtype=np.uint8,
        )
        paulis, phases = self.propagate_paulis(self.all_members, all_paulis)
        indexer = np.array([4**idx for idx in range(self.num_qubits)])[::-1]
        return paulis, phases, indexer

    @cached_property_by_dim
    @abc.abstractmethod
    def _parent_orbits(self):
        pass

    def bootstrap(self):
        # trigger table construction before patching, otherwise we will get infinite recursion
        self._full_dot_table  # pylint: disable=pointless-statement
        self._full_inv_table  # pylint: disable=pointless-statement
        self._full_propagate_paulis_table  # pylint: disable=pointless-statement
        self.dot = self._full_dot
        self.inv = self._full_inv
        # self.propagate_paulis = self._full_propagate_paulis
        return self

    @property
    def bounds(self):
        return tuple(map(len, self._parent_orbits))

    def dot(self, lhs, rhs):
        product = self._parent.dot(self._all_parent_members[lhs], self._all_parent_members[rhs])
        return self._lookup_parent_members(product)

    def from_operation(self, operation):
        return self._lookup_parent_members(self._parent.from_operation(operation))

    def inv(self, members):
        return self._lookup_parent_members(self._parent.inv(self._all_parent_members[members]))

    def permute(self, members, permutations):
        members = self._all_parent_members[members]
        members = self._parent.permute(members, permutations)
        return self._lookup_parent_members(members)

    def propagate_paulis(self, members, paulis, phases=np.array([0])):
        members = self._all_parent_members[members]
        return self._parent.propagate_paulis(members, paulis, phases)

    def verify(self, shape=(10, 100), seed=None):
        lhs = self.sample(shape, seed)
        rhs = self.sample(shape, seed)

        # verify that products are correct
        products = self._parent.dot(self._all_parent_members[lhs], self._all_parent_members[rhs])
        assert self.equal(self._lookup_parent_members(products), self.dot(lhs, rhs))

        # verify that inverses are correct
        inverses = self._parent.inv(self._all_parent_members[lhs])
        assert self.equal(self._lookup_parent_members(inverses), self.inv(lhs))

    def __repr__(self):
        return f"{type(self).__name__}()"
