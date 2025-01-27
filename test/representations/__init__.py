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
Test circuit representation implementations
"""

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

import twirly as tw


class CircuitRepresentationTestCase:
    """Mixin for testing methods common to all ``CircuitRepresentation`` implementations."""

    def test_append_instructions(self):
        # generate random group members to fill a circuit of shape (depth + 1) x (num_members * n)
        # where n is the number of qubits per group member, as defined by _twirling_group
        members = self.rep.twirling_group.sample(
            (self.num_members, self.depth + 1, self.num_randomizations)
        )
        # we will use the last layer of the circuit to invert the previous layers
        members[:, -1, ...] = self.rep.twirling_group.id_member()

        # use the same loop to accumulate the group member products, and also build a parametered
        # circuit of the specified shape
        qreg = QuantumRegister(self.num_members * self.rep.injection.codomain.num_qubits)
        qubits = tw.base.Subsystems.from_iterable(qreg, self.rep.twirling_group.num_qubits)
        base_circuit = QuantumCircuit(qreg)
        circuits = tw.base.CircuitBundle(base_circuit)
        for idx in range(self.depth):
            members[:, -1, ...] = self.rep.injection.domain.dot(
                members[:, idx, ...], members[:, -1, ...]
            )
            self.rep.append_instructions(circuits, qubits, members[:, idx, ...])
        members[:, -1, ...] = self.rep._injection.domain.inv(members[:, -1, ...])
        self.rep.append_instructions(circuits, qubits, members[:, -1, ...])

        self.assertEqual(len(circuits), self.num_randomizations)

        # ensure that the circuit simulates to the identity
        backend = AerSimulator(method="unitary")
        for circuit in circuits:
            circuit.save_state()
            unitary = backend.run(circuit).result().get_unitary()
            overlap = np.abs(np.trace(unitary)) / (
                self.rep.injection.codomain.dim**self.num_members
            )
            self.assertAlmostEqual(overlap, 1)

        return circuits


class TemplateRepresentationTestCase(CircuitRepresentationTestCase):
    """Mixin for testing methods common to all ``TemplateRepresentation`` implementations."""

    def test_append_instructions(self):
        circuits = super().test_append_instructions()

        self.assertTrue(circuits.has_single_base)
        self.assertEqual(
            circuits.base_circuit.num_parameters,
            self.num_members * (self.depth + 1) * sum(self.rep.num_params_per_member),
        )

    def test_generate_template(self):
        qreg = QuantumRegister(self.rep.twirling_group.num_qubits)
        num_parameters = sum(self.rep.num_params_per_member)
        parameters = iter(Parameter(f"p{idx}") for idx in range(num_parameters))
        instructions = self.rep.generate_template(qreg, parameters)

        # ensure that the instructions use the parameters
        circuit = QuantumCircuit(qreg)
        for instruction in instructions:
            circuit.append(instruction)
        self.assertEqual(circuit.num_parameters, num_parameters)

        # ensure that generate_template consumed     exactly the number of parameters it promises
        with self.assertRaises(StopIteration):
            next(parameters)

    def test_parameter_values(self):
        members = self.rep.twirling_group.sample((1, 2, 3))
        vals = self.rep.parameter_values(members)

        self.assertEqual(vals.shape[:3], (1, 2, 3))
        self.assertEqual(np.prod(vals.shape[3:]), sum(self.rep.num_params_per_member))
