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
Test groups/haar.py
"""

from functools import reduce

import numpy as np

import twirly as tw

from .. import TwirlyTestCase


def multidim_kron(*arrs):
    d1, d2 = 1, 1
    shape = None
    flat_arrs = []
    for arr in arrs:
        d1 *= arr.shape[-2]
        d2 *= arr.shape[-1]
        if shape is None:
            shape = arr.shape[:-2]
        else:
            assert arr.shape[:-2] == shape
        flat_arrs.append(arr.reshape(-1, *arr.shape[-2:]))

    ret = np.empty(shape + (d1, d2), dtype=complex)
    flat_ret = ret.reshape((-1, d1, d2))
    for member, elems in zip(flat_ret, zip(*flat_arrs)):
        member[...] = reduce(np.kron, elems)
    return ret


class HaarUnitaryTestCase(TwirlyTestCase):
    def test_equal(self):
        h2 = tw.HaarUnitary(2)
        arr1 = h2.sample((5, 4))
        arr2 = h2.sample((5, 4))

        self.assertTrue(h2.equal(arr1, arr1))
        self.assertTrue(h2.equal(arr1[:, 0, ...], np.broadcast_to(arr1[:, 0, ...], (5, 5, 4, 4))))
        self.assertFalse(h2.equal(arr1, arr1[:, 0, ...]))
        self.assertFalse(h2.equal(arr1, arr2))

    def test_permute(self):
        h1 = tw.HaarUnitary(1)
        h4 = tw.HaarUnitary(4)

        a, b, c, d = h1.sample((4, 2, 3))
        members = multidim_kron(a, b, c, d)

        """Performs a subsystem permutation of each member.

        Permutations are specified in the same format as ``numpy.transpose``. For example, the
        permutation ``[2, 0, 1]`` translates to "make the current 2nd substem the new 0th subsystem,
        the current 0th subsystem the new 1st subsystem, and the current 1st substem the new 2nd
        subsystem".

        Args:
            members: A member array.
            permutations: An integer array whose first dimensions are broadcastable to the member
                array shape of ``members``, and whose last dimension is equal to
                :attr:`~num_qubits`, containing the subsystem permutation for the corresponding
                member.
        Returns:
            A new member array with permuted subsystems.
        """

        # trivial permutation
        permuted = h4.permute(members, np.array([0, 1, 2, 3]))
        self.assertTrue(h4.equal(members, permuted))

        # uniform permutation
        permuted = h4.permute(members, np.array([2, 0, 1, 3]))
        self.assertTrue(h4.equal(multidim_kron(c, a, b, d), permuted))

        # uniform permutation without broadcasting
        permuted = h4.permute(members, np.array([[[2, 0, 1, 3]]]))
        self.assertTrue(h4.equal(multidim_kron(c, a, b, d), permuted))

        # different permutations along second axis
        permuted = h4.permute(members, np.array([[0, 1, 2, 3], [2, 0, 1, 3], [0, 2, 1, 3]]))
        e0 = members[:, 0, None, ...]
        e1 = multidim_kron(
            c[:, 1, None, ...],
            a[:, 1, None, ...],
            b[:, 1, None, ...],
            d[:, 1, None, ...],
        )
        e2 = multidim_kron(
            a[:, 2, None, ...],
            c[:, 2, None, ...],
            b[:, 2, None, ...],
            d[:, 2, None, ...],
        )
        self.assertTrue(h4.equal(np.concatenate([e0, e1, e2], axis=1), permuted))

        #  different permutations along first axis
        permuted = h4.permute(members, np.array([[[0, 1, 2, 3]], [[2, 0, 1, 3]]]))
        e0 = members[0, None, ...]
        e1 = multidim_kron(c[1, None, ...], a[1, None, ...], b[1, None, ...], d[1, None, ...])
        self.assertTrue(h4.equal(np.concatenate([e0, e1], axis=0), permuted))
