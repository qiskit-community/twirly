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
Haar uniform unitary group representations
"""

import numpy as np
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import RZGate, SXGate

from ..base import CachedTemplateRepresentation
from ..groups import HaarUnitary
from ..utils import shape_tuple


class Unitary1QZXZXZ(CachedTemplateRepresentation):
    def __init__(self, twirling_group=None):
        super().__init__(twirling_group or HaarUnitary(1))
        # reduce total instance count by defining here
        self._sx = SXGate()

    @property
    def num_params_per_member(self):
        return (3,)

    def generate_template(self, qubits, parameters):
        parameter = iter(parameters)
        for qubit in qubits:
            qubit = (qubit,)
            yield CircuitInstruction(RZGate(next(parameter)), qubit)
            yield CircuitInstruction(self._sx, qubit)
            yield CircuitInstruction(RZGate(next(parameter)), qubit)
            yield CircuitInstruction(self._sx, qubit)
            yield CircuitInstruction(RZGate(next(parameter)), qubit)

    def parameter_values(self, members):
        # do an euler decomposition from the 2x2 unitaries
        members = self._injection.apply(members)
        phase = (
            members[..., 0, 0] * members[..., 1, 1] - members[..., 0, 1] * members[..., 1, 0]
        ) ** -0.5
        phi_plus_lambda = np.angle(phase * members[..., 1, 1])
        phi_minus_lambda = np.angle(phase * members[..., 1, 0])
        values = np.empty(shape_tuple(members.shape[:-2], self.num_params_per_member), dtype=float)
        values[..., 2] = phi_plus_lambda + phi_minus_lambda
        values[..., 1] = np.pi - 2 * np.arctan2(
            np.abs(members[..., 1, 0]), np.abs(members[..., 0, 0])
        )
        values[..., 0] = phi_plus_lambda - phi_minus_lambda - np.pi
        # restrict all angles to (-pi, pi]
        return -np.remainder(-values + np.pi, 2 * np.pi) + np.pi
