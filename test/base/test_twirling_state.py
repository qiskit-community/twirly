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
Test base/twirling_state.py
"""

import numpy as np
from qiskit.circuit import QuantumCircuit, Qubit

import twirly as tw
from twirly.base import IndexedMemberArray, Subsystems

from .. import TwirlyTestCase


class SubsystemsTestCase(TwirlyTestCase):
    def test_init_raises(self):
        with self.assertRaisesRegex(tw.TwirlingError, "num_qubits must be provided"):
            Subsystems([])
        with self.assertRaisesRegex(tw.TwirlingError, "cannot be ragged"):
            Subsystems([[Qubit()], [Qubit(), Qubit()]])

    def test_properties(self):
        subsys = Subsystems([], num_qubits=4)
        self.assertEqual(subsys.num_qubits, 4)
        self.assertEqual(subsys.qubits, ())
        self.assertEqual(subsys.num_subsystems, 0)
        self.assertEqual(subsys.num_total_qubits, 0)
        self.assertEqual(subsys.unordered_map, {})

        qubits = [[Qubit(), Qubit()], [Qubit(), Qubit()], [Qubit(), Qubit()]]
        subsys = Subsystems(qubits)
        self.assertEqual(subsys.num_qubits, 2)
        self.assertEqual(subsys.qubits, tuple(map(tuple, qubits)))
        self.assertEqual(subsys.num_subsystems, 3)
        self.assertEqual(subsys.num_total_qubits, 6)
        self.assertEqual(subsys.unordered_map, {frozenset(qubits[idx]): idx for idx in range(3)})

    def test_eq(self):
        qubits = [[Qubit(), Qubit()], [Qubit(), Qubit()]]
        self.assertEqual(Subsystems(qubits), Subsystems(qubits))
        self.assertNotEqual(Subsystems(qubits), Subsystems(qubits[:1]))
        self.assertNotEqual(Subsystems(qubits), Subsystems([qubits[0] + qubits[1]]))
        self.assertNotEqual(Subsystems(qubits), Subsystems([qubits[0], [Qubit(), Qubit()]]))

    def test_iter(self):
        qubits = [[Qubit(), Qubit()], [Qubit(), Qubit()]]
        for idx, elem in enumerate(Subsystems(qubits)):
            self.assertEqual(elem, tuple(qubits[idx]))

    def test_get_item(self):
        qubits = q0, q1, q2 = [
            (Qubit(), Qubit()),
            (Qubit(), Qubit()),
            (Qubit(), Qubit()),
        ]
        self.assertEqual(Subsystems(qubits)[2], q2)
        self.assertEqual(Subsystems(qubits)[1:], (q1, q2))
        self.assertEqual(Subsystems(qubits)[0, 2], (q0, q2))
        self.assertEqual(Subsystems(qubits)[[0, 2]], (q0, q2))

    def test_len(self):
        self.assertEqual(len(Subsystems([], num_qubits=2)), 0)
        self.assertEqual(len(Subsystems([[Qubit(), Qubit()]])), 1)

    def test_flat_iter(self):
        qubits = [Qubit(), Qubit(), Qubit(), Qubit()]
        subsys = Subsystems([qubits[:2], qubits[2:]])
        for q0, q1 in zip(qubits, subsys.flat_iter()):
            self.assertEqual(q0, q1)

    def test_from_iterable(self):
        qubits = [Qubit(), Qubit(), Qubit(), Qubit(), Qubit(), Qubit()]

        subsys = Subsystems.from_iterable(qubits, num_qubits=2)
        self.assertEqual(subsys, Subsystems([qubits[:2], qubits[2:4], qubits[4:]]))

        subsys = Subsystems.from_iterable(qubits, num_qubits=3)
        self.assertEqual(subsys, Subsystems([qubits[:3], qubits[3:]]))

    def test_reshape(self):
        qubits = [Qubit(), Qubit(), Qubit(), Qubit(), Qubit(), Qubit()]
        subsys = Subsystems([qubits[:3], qubits[3:]])

        self.assertEqual(subsys.reshape(1), Subsystems([[qubit] for qubit in qubits]))
        self.assertEqual(subsys.reshape(2), Subsystems([qubits[:2], qubits[2:4], qubits[4:]]))
        self.assertEqual(subsys.reshape(3), subsys)
        self.assertEqual(subsys.reshape(6), Subsystems([qubits]))


class IndexedMemberArrayTestCase(TwirlyTestCase):
    def assertIMAEqual(self, lhs: IndexedMemberArray, rhs: IndexedMemberArray):
        assert lhs.qubits == rhs.qubits
        assert lhs.twirling_group == rhs.twirling_group
        assert lhs.twirling_group.equal(lhs.members, rhs.members)

    def test_properties(self):
        qubits = [[Qubit()], [Qubit()], [Qubit()], [Qubit()]]
        haar = tw.HaarUnitary(1)
        members = haar.sample((4, 15))
        ima = IndexedMemberArray(qubits, haar, members)

        self.assertEqual(ima.qubits, Subsystems(qubits))
        self.assertEqual(ima.twirling_group, haar)
        self.assertTrue(haar.equal(members, ima.members))
        self.assertEqual(ima.full_shape, (4, 15))
        self.assertEqual(ima.ndim, 1)
        self.assertEqual(ima.num_subsystems, 4)
        self.assertEqual(ima.num_qubits, 1)
        self.assertEqual(ima.shape, (15,))

    def test_iter(self):
        qubits = [[Qubit()], [Qubit()], [Qubit()], [Qubit()]]
        haar = tw.HaarUnitary(1)
        members = haar.sample((4, 15))
        ima_iter = iter(IndexedMemberArray(qubits, haar, members))

        self.assertEqual(next(ima_iter), Subsystems(qubits))
        self.assertEqual(next(ima_iter), haar)
        self.assertTrue(haar.equal(next(ima_iter), members))
        self.assertRaises(StopIteration, next, ima_iter)

    def test_from_instructions(self):
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cz(2, 3)

        # we treat testing TwirlingGroup.from_operation() as out-of-scope
        ima = IndexedMemberArray.from_instructions(tw.Clifford2Q(), circuit)
        self.assertEqual(ima.qubits, Subsystems([circuit.qregs[0][:2], circuit.qregs[0][2:]]))
        self.assertEqual(ima.twirling_group, tw.Clifford2Q())
        self.assertEqual(ima.shape, ())

    def test_join(self):
        haar = tw.HaarUnitary(1)
        ima0 = IndexedMemberArray([[Qubit()]], haar, haar.sample((1, 2, 3)))
        ima1 = IndexedMemberArray([[Qubit()], [Qubit()], [Qubit()]], haar, haar.sample((3, 2, 3)))
        ima2 = IndexedMemberArray([[Qubit()], [Qubit()]], haar, haar.sample((2, 2, 3)))

        ima = IndexedMemberArray.join([ima0, ima1, ima2])
        self.assertEqual(ima.qubits, Subsystems(ima0.qubits[:] + ima1.qubits[:] + ima2.qubits[:]))
        self.assertEqual(ima.twirling_group, haar)
        self.assertEqual(ima.full_shape, (6, 2, 3))
        self.assertTrue(haar.equal(ima.members[:1], ima0.members))
        self.assertTrue(haar.equal(ima.members[1:4], ima1.members))
        self.assertTrue(haar.equal(ima.members[4:], ima2.members))

    def test_permute(self):
        # we use 3-qubit group so that it is possible to test beyond order-2 permutations
        haar = tw.HaarUnitary(3)
        qubits = s0, s1, s2, s3 = [[Qubit() for _ in range(3)] for _ in range(4)]

        # manually make all unitaries separable to make checking permutations transparent
        tensor_members = tw.HaarUnitary(1).sample((4, 2, 5, 3))
        members = haar.empty((4, 2, 5))
        for (a, b, c), m in zip(tensor_members.reshape((40, 3, 2, 2)), members.reshape((40, 8, 8))):
            m[...] = np.kron(np.kron(a, b), c)
        ima = IndexedMemberArray(qubits, haar, members)

        # trivial permutation
        self.assertIMAEqual(ima.permute(Subsystems([s0, s1, s2, s3])), ima)

        # permute only subsystem order
        permuted = ima.permute(Subsystems([s3, s0, s1, s2]))
        expected = IndexedMemberArray([s3, s0, s1, s2], haar, members[[3, 0, 1, 2], ...])
        self.assertIMAEqual(permuted, expected)

        # permute only qubit order
        permuted_qubits = [[qs[idx] for idx in [2, 0, 1]] for qs in qubits]
        permuted = ima.permute(Subsystems(permuted_qubits))
        expected_members = haar.permute(members, np.array([1, 2, 0]))
        self.assertIMAEqual(permuted, IndexedMemberArray(permuted_qubits, haar, expected_members))

        # permute both subsystem order and qubit order
        permuted_qubits = [[qs[idx] for idx in [2, 0, 1]] for qs in Subsystems([s3, s0, s1, s2])]
        permuted = ima.permute(Subsystems(permuted_qubits))
        expected_members = haar.permute(members[[3, 0, 1, 2], ...], np.array([1, 2, 0]))
        self.assertIMAEqual(permuted, IndexedMemberArray(permuted_qubits, haar, expected_members))

    def test_replace(self):
        qubits = [[Qubit()], [Qubit()], [Qubit()], [Qubit()]]
        haar = tw.HaarUnitary(1)
        members = haar.sample((4, 15))
        ima = IndexedMemberArray(qubits, haar, members)

        new_qubits = [[Qubit()], [Qubit()], [Qubit()], [Qubit()]]
        self.assertIMAEqual(
            ima.replace(qubits=new_qubits),
            IndexedMemberArray(new_qubits, haar, members),
        )

        new_members = haar.sample((4, 2))
        self.assertIMAEqual(
            ima.replace(members=new_members),
            IndexedMemberArray(qubits, haar, new_members),
        )

        class HaarSubclass(tw.HaarUnitary):
            pass

        new_haar = HaarSubclass(1)
        self.assertIMAEqual(
            ima.replace(twirling_group=new_haar),
            IndexedMemberArray(qubits, new_haar, members),
        )

    def test_split(self):
        qubits = [[Qubit(), Qubit()] for _ in range(6)]
        haar = tw.HaarUnitary(2)
        members = haar.sample((6, 15))
        ima = IndexedMemberArray(qubits, haar, members)

        split_qubits = qubits[::2]
        # rearange one of the subsystems to test the promise of order-irrelevance
        split_qubits[1] = split_qubits[1][::-1]
        ima0, ima1 = ima.split(Subsystems(split_qubits))
        self.assertIMAEqual(ima0, IndexedMemberArray(qubits[::2], haar, members[::2, ...]))
        self.assertIMAEqual(ima1, IndexedMemberArray(qubits[1::2], haar, members[1::2, ...]))

    def test_subsys_slice(self):
        qubits = [[Qubit(), Qubit()] for _ in range(6)]
        haar = tw.HaarUnitary(2)
        members = haar.sample((6, 15))
        ima = IndexedMemberArray(qubits, haar, members)

        # slice object
        expected = IndexedMemberArray(qubits[::2], haar, members[::2, ...])
        self.assertIMAEqual(ima.subsys_slice(slice(None, None, 2)), expected)

        # fancy slice
        expected = IndexedMemberArray([qubits[0], qubits[4]], haar, members[[0, 4], ...])
        self.assertIMAEqual(ima.subsys_slice([0, 4]), expected)
