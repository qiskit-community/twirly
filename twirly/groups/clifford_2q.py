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
Clifford twirling group for a two qubits
"""

from itertools import product

import numpy as np
from qiskit.quantum_info import Clifford

from ..utils import cached_property_by_dim, shape_tuple
from .clifford import SmallCliffordGroup, UniformClifford
from .clifford_1q import Clifford1Q


def _embed(members, pos):
    c2 = UniformClifford(2)
    mask = np.array([[6, 8, 9], [16, 18, 19]]) if pos else np.array([[0, 2, 4], [10, 12, 14]])
    ret = c2.id_member(members.shape[:-2])
    ret.reshape(members.shape[0], -1)[..., mask] = members
    return ret


class Clifford2Q(SmallCliffordGroup):
    r"""The two-qubit Clifford group."""

    def __init__(self):
        super().__init__(2)
        self._c1 = Clifford1Q()

    @cached_property_by_dim
    def _basis_changes(self):
        # the 9 basis changes to turn CNOT into each of the 9 non-trivial generalized CNOTs
        hs = Clifford.from_label("H") @ Clifford.from_label("S")
        hs_orbit = UniformClifford(1).orbit_from_clifford(hs)
        return self._parent.outer(_embed(hs_orbit, 1), _embed(hs_orbit, 0), ravel=True)

    @cached_property_by_dim
    def _coset_mult_table(self):
        gen_cnots = self._parent_orbits[1]
        table = self._parent.outer(gen_cnots, gen_cnots)
        return self.unpack(self._lookup_parent_members(table))

    @cached_property_by_dim
    def _coset_swap_table(self):
        swap, gen_cnots = self._parent_orbits[:2]
        table = self._parent.dot(self._parent.outer(swap, gen_cnots), swap[:, None, ...])
        return self.unpack(self._lookup_parent_members(table))[..., 1]

    @cached_property_by_dim
    def _coset_perm0_table(self):
        gen_cnots, c1 = self._parent_orbits[1:3]
        table = self._parent.outer(c1, gen_cnots)
        table = self._parent.dot(table, self._parent.inv(c1)[:, None, ...])
        return self.unpack(self._lookup_parent_members(table))[..., [1, 3]]

    @cached_property_by_dim
    def _coset_perm1_table(self):
        gen_cnots, _, c1 = self._parent_orbits[1:4]
        table = self._parent.outer(c1, gen_cnots)
        table = self._parent.dot(table, self._parent.inv(c1)[:, None, ...])
        return self.unpack(self._lookup_parent_members(table))[..., [1, 2]]

    @cached_property_by_dim
    def _parent_orbits(self):
        ofc = self._parent.orbit_from_clifford
        swap = [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0]]
        cx = [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 0]]

        generalized_cxs = self._parent.dot(
            self._basis_changes, np.array([cx], dtype=self._parent.dtype)
        )
        generalized_cxs = self._parent.dot(generalized_cxs, self._parent.inv(self._basis_changes))
        generalized_cxs = np.concatenate([self._parent.id_member(1), generalized_cxs])

        c1 = self._c1._all_parent_members
        return (ofc(swap), generalized_cxs, _embed(c1, 1), _embed(c1, 0))

    def dot(self, lhs, rhs):
        shape = np.broadcast_shapes(self.member_array_shape(lhs), self.member_array_shape(rhs))
        ret = np.empty(shape_tuple(shape, len(self.bounds)), dtype=self.dtype)
        lhs = self.unpack(lhs)
        rhs = self.unpack(rhs)
        ret[...] = lhs

        # swap LHS single-qubit cliffords conditional on RHS SWAP
        ret[..., [2, 3]] += rhs[..., 0, None] * (lhs[..., [3, 2]] - lhs[..., [2, 3]])
        # swap LHS generalized-cnot conditional on RHS SWAP
        ret[..., 1] = self._coset_swap_table[rhs[..., 0], lhs[..., 1]]
        # XOR the swap bits together
        ret[..., 0] += rhs[..., 0]

        # determine what happens to the RHS coset and single-qubit cliffords when we move the LHS
        # single-qubit cliffords past the RHS coset
        c1_past_rhs_coset = self._coset_perm0_table[ret[..., 2], rhs[..., 1], :]
        c2_past_rhs_coset = self._coset_perm1_table[ret[..., 3], c1_past_rhs_coset[..., 0]]

        # now see which swap, new coset, additional single-qubit cliffords we get in coset
        # multiplication
        csm = self._coset_mult_table[ret[..., 1], c2_past_rhs_coset[..., 0]]

        # incorporate the possible additional swap and new coset
        ret[..., 0] += csm[..., 0]
        ret[..., 1] = csm[..., 1]

        # all single-qubit cliffords have been moved to the right: join them together
        m1 = self._c1._full_dot_table
        ret[..., 2] = m1[m1[m1[csm[..., 2], c2_past_rhs_coset[..., 1]], ret[..., 2]], rhs[..., 2]]
        ret[..., 3] = m1[m1[m1[csm[..., 3], ret[..., 3]], c1_past_rhs_coset[..., 1]], rhs[..., 3]]

        ret[..., 0] %= 2
        return self.pack(ret)

    def inv(self, members):
        ret = self.unpack(members)
        # invert the single-qubit cliffords
        ret[..., [2, 3]] = self._c1._full_inv_table[ret[..., [2, 3]]]
        # swap generalized-cnots conditional on the input SWAP flag
        ret[..., 1] = self._coset_swap_table[ret[..., 0], ret[..., 1]]
        # swap the single-qubit cliffords conditional on the input SWAP flag
        ret[..., [2, 3]] += ret[..., 0, None] * (ret[..., [3, 2]] - ret[..., [2, 3]])

        # determine what happens to the coset and single-qubit cliffords when we move the (now
        # inverted) single-qubit cliffords past the coset
        c1_past_rhs_coset = self._coset_perm0_table[ret[..., 2], ret[..., 1], :]
        c2_past_rhs_coset = self._coset_perm1_table[ret[..., 3], c1_past_rhs_coset[..., 0]]

        # join sinle qubit gates together and update the coset
        ret[..., 1] = c2_past_rhs_coset[..., 0]
        ret[..., 2] = self._c1._full_dot_table[c2_past_rhs_coset[..., 1], ret[..., 2]]
        ret[..., 3] = self._c1._full_dot_table[ret[..., 3], c1_past_rhs_coset[..., 1]]

        return self.pack(ret)

    def permute(self, members, permutations):
        ret = self.unpack(members)
        # each permutation is either [0,1] or [1,0], so the first entry serves as a swap bool
        do_swap = permutations[..., 0]
        # swap the single-qubit cliffords conditional on do_swap
        ret[..., [2, 3]] += do_swap[..., None] * (ret[..., [3, 2]] - ret[..., [2, 3]])
        # swap generalized-cnots conditional on do_swap
        ret[..., 1] = self._coset_swap_table[do_swap, ret[..., 1]]
        return self.pack(ret)

    @cached_property_by_dim
    def _coset_propagate_paulis_table(self):
        # find out what each of the 20 first coset representatives do to all 16 paulis
        all_cosets = self._all_parent_members[np.arange(20, dtype=self.dtype) * 576]
        all_paulis = np.array(
            list(map(list, product([0, 1, 2, 3], repeat=self.num_qubits))),
            dtype=np.uint8,
        )
        paulis, phases = self._parent.propagate_paulis(all_cosets, all_paulis)
        return paulis, phases

    def propagate_paulis2(self, members, paulis, phases=np.array([0])):
        # first propagate the paulis through the single-qubit Cliffords, accumulating phasse
        unpacked = self.unpack(members)
        paulis0, phases0 = self._c1.propagate_paulis(unpacked[..., 2], paulis[..., 1:], phases)
        paulis1, phases1 = self._c1.propagate_paulis(unpacked[..., 3], paulis[..., :1], phases0)

        # and then use a small 20x9 lookup tables to get past the generalized CNOT + SWAP cosets
        paulis_lookup, phases_lookup = self._coset_propagate_paulis_table
        members = members[..., None] // (24 * 24)
        idxs = (4 * paulis1 + paulis0)[..., 0]
        return paulis_lookup[members, idxs], phases_lookup[members, idxs] ^ phases1


class A5(SmallCliffordGroup):
    def __init__(self):
        super().__init__(2)

    @cached_property_by_dim
    def _parent_orbits(self):
        ofc = self._parent.orbit_from_clifford
        c5 = [[1, 0, 1, 0, 1], [1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [0, 1, 0, 1, 1]]
        c2 = [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 1, 1, 1, 1]]
        s3 = [[1, 0, 1, 0, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 1], [0, 1, 0, 1, 1]]
        s2 = [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        return (ofc(c5), ofc(c2), ofc(s3), ofc(s2))


class G29D(SmallCliffordGroup):
    def __init__(self):
        super().__init__(2)
        self._a5 = A5().bootstrap()

    @cached_property_by_dim
    def _conj_table(self):
        a5, paulis = self._parent_orbits
        table = self._parent.dot(self._parent.outer(self._parent.inv(a5), paulis), a5[:, None, ...])
        return self._lookup_parent_members(table)

    @cached_property_by_dim
    def _parent_orbits(self):
        c2 = Clifford2Q()
        p0 = c2._parent_orbits[3][:4]
        p1 = c2._parent_orbits[2][:4]
        return (self._a5._all_parent_members, self._parent.outer(p1, p0, ravel=True))

    @cached_property_by_dim
    def _pauli_dot_table(self):
        paulis = self._parent_orbits[1]
        return self._lookup_parent_members(self._parent.outer(paulis, paulis))

    def dot(self, lhs, rhs):
        shape = np.broadcast_shapes(self.member_array_shape(lhs), self.member_array_shape(rhs))
        ret = np.empty(shape_tuple(shape, len(self.bounds)), dtype=self.dtype)
        lhs = self.unpack(lhs)
        rhs = self.unpack(rhs)

        ret[..., 0] = self._a5._full_dot_table[lhs[..., 0], rhs[..., 0]]
        ret[..., 1] = self._pauli_dot_table[self._conj_table[rhs[..., 0], lhs[..., 1]], rhs[..., 1]]
        return self.pack(ret)

    def inv(self, members):
        ret = self.unpack(members)
        ret[..., 0] = self._a5._full_inv_table[ret[..., 0]]
        ret[..., 1] = self._conj_table[ret[..., 0], ret[..., 1]]
        return self.pack(ret)
