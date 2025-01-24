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
Two-qubit Clifford group maps
"""

import numpy as np

from qiskit.circuit import Operation
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Clifford

from ..base import GroupMap, ENUMERATED_DTYPE
from ..groups import Clifford1Q, Clifford2Q
from ..utils import cached_property_by_dim


def _cx_conversion(gcx: int):
    r"""Return the single-qubit Clifford gates that conjugate the given generalized-CX into CX.

    It holds that :math:`CX = B_0^\dagger \otimes B_1^\dagger GCX B_0 \otimes B_1`, where
    :math:`B_0` and :math:`B_1` are the two outputs of this function.
    """
    # recall that Clifford2Q's convention is that 0 <= gcx < 10, and that gcx-1 corresponds
    # to CX conjugated by some power of HS on qubit 0, and some power of HS on qubit 1. we
    # can find these powers with "%3" and "// 3" respectively. Next, recall that HS is the
    # MSB in our convention for the lexicographical ordering of Clifford1Q(), following H,
    # X, and Z, each of which have order 2, hence the multiplication by 8 to retrieve the
    # correct index
    b0 = 8 * ((gcx - 1) % 3)
    b1 = 8 * ((gcx - 1) // 3)
    return np.array([b0, b1], dtype=ENUMERATED_DTYPE)


class Clifford2QToInterleavedCXLike(GroupMap):
    r"""Performs a fixed-depth CX-like decomposition of :class:`~Clifford2Q` group members.

    Here, a CX-like gate is any two-qubit Clifford gate that is equivalent to the CX gate up to
    multiplication on either side by single-qubit Cliffords.

    The values returned by :meth:`~apply` dictate the members of :class:`~Clifford1Q` to interleave
    between three (implicit) CX-like operations. This is stored in the last two dimensions, with
    shape ``(2, 4)``, where the values ``[..., 0, :]`` and ``[..., 1, :]`` respectively record the
    single-qubit Cliffords on qubit 0 and 1, and where the last index is in operator order (i.e.
    reversed time order).
    """

    def __init__(self, operation: Operation = CXGate()):
        super().__init__(Clifford2Q(), Clifford1Q())

        # the provided operation is CX-like iff gcx > 0 (i.e. not the trivial generalized CNOT)
        # and if sw==0 (i.e. doesn't have a swap component, which would be iSWAP-like for gcx>0)
        sw, gcx, c1, c0 = self.domain.unpack(self.domain.from_operation(operation))
        if not (gcx > 0 and not sw):
            raise ValueError(f"The provided operation {operation} is not CX-like.")

        # the following calculations guarantee the identity CX = lhs_mod @ G @ rhs_mod, where G is
        # the CX-like of this instance
        b = _cx_conversion(gcx).reshape(-1, 1)
        self._lhs_mod = self.codomain.inv(b)
        self._rhs_mod = self.codomain.dot(b, np.array([[c0], [c1]], dtype=self.codomain.dtype))

    def _constant_depth_cx_decomp(self, member):
        sw, gcx, c1, c0 = self.domain.unpack(member)
        decomp = np.empty((4, 2), dtype=self.codomain.dtype)

        # start by populating the output with this member's coset
        decomp[3, :] = [c0, c1]

        # below we handle the four cases of decompostion due to the bool pair (sw, gcx>0)
        if gcx:
            b = _cx_conversion(gcx)
            # first, combine the GCX transformation with the c1, c0 in the first (temporal layer)
            decomp[3, :] = self.codomain.dot(self.codomain.inv(b), decomp[3, :])
            if sw:
                # iSWAP-like case: we have a gate of the form
                # SWAP @ (b0, b1) @ CX @ (b0_inv, b1_inv) @ (c0, c1). our strategy is:
                # 1. combine (b0_inv, b1_inv) @ (c0, c1) in layer 3 (already done outside of "if")
                # 2. move (b0, b1) past swap by reversing their order; put b[::-1] in layer 0
                # 3. implement SWAP @ CX by inserting appropriate gates in layers 1-3. in
                #    particular, it can be verified that [[4,8], [0,6], [16, 12]] does the trick
                decomp[0, :] = b[::-1]
                decomp[3, :] = self.codomain.dot(
                    np.array([16, 12], dtype=ENUMERATED_DTYPE), decomp[3, :]
                )
                decomp[1:3, :] = [[4, 8], [0, 6]]

            else:
                # CX-like case: we complete the conjugate side of the GCX transformation, and
                # all the other single-qubit gates can be 0 because CX^2 = id
                decomp[2, :] = b
                decomp[:2, :] = 0
        elif sw:
            # SWAP-like case: flip the direction of the central CX with hadamards (i.e. member 4)
            decomp[1:3, :] = 4
            decomp[0, :] = 0
        else:
            # trivial case: implementation-wise, not as trivial as the CX- or SWAP-like cases; need
            # to implement an id out of 3 CXs. to do this, use the fact that
            # CX = (0, 21) @ CX @ (0, 6) @ CX @ (20, 8), so that this times CX is id.
            decomp[0, :] = 0
            decomp[1:3, :] = [[0, 21], [0, 6]]
            decomp[3, :] = self.codomain.dot(
                np.array([20, 8], dtype=self.codomain.dtype), decomp[3, :]
            )

        return decomp[::-1, :].T

    @cached_property_by_dim
    def _all_cx_decomps(self):
        decomp_func = np.vectorize(self._constant_depth_cx_decomp, signature="()->(k,l)")
        return decomp_func(self.domain.all_members)

    def apply(self, members):
        decomps = self._all_cx_decomps[members]

        # for one particular entry D=[*idx, :, :].T of decomps, we have the corresponding
        # decomposition D[0] @ CX @ D[1] @ CX @ D[2] @ CX @ D[3]. we can replace each CX with the G
        # operation using D[0] @ lhs @ G @ rhs @ D[1] @ lhs @ G @ rhs @ D[2] @ lhs @ G @ rhs @ D[3],
        # hence the following modifications

        # TODO: figure out why LHS and RHS need to be swapped from the above description for it to
        # work
        decomps[..., 1:] = self.codomain.dot(self._lhs_mod, decomps[..., 1:])
        decomps[..., :-1] = self.codomain.dot(decomps[..., :-1], self._rhs_mod)
        return decomps

    def codomain_shapes(self, input_shape):
        return (input_shape + (2, 4),)

    def domain_shapes(self, output_shape):
        return (output_shape[:-2],)

    def _verify(self, member: int, operation: Operation = CXGate()):
        m = lambda x: Clifford(self.codomain._all_parent_members[x])
        c = Clifford.from_label("II")
        for a, b in zip(*self.apply(member).T):
            c = Clifford(operation) @ (m(b) ^ m(a)) @ c
        c = Clifford(operation) @ c
        d = Clifford(self.domain._all_parent_members[member])
        if c != d:
            print(c)
            print(d)
