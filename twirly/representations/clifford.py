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
Clifford group circuit representations
"""

from typing import Optional

from qiskit.circuit import CircuitInstruction, Barrier, Operation
from qiskit.circuit.library import CXGate, RZGate, SXGate

from ..base import TemplateRepresentation
from ..groups import Clifford2Q
from ..maps import Clifford1QIntoHaarUnitary, Clifford2QToInterleavedCXLike
from .haar import Unitary1QZXZXZ


class Clifford1QZXZXZ(Unitary1QZXZXZ):
    def __init__(self):
        super().__init__(Clifford1QIntoHaarUnitary())


class Clifford2QCXLikeRepresentation(TemplateRepresentation):
    def __init__(
        self,
        operation: Operation = CXGate(),
        clifford_1q_rep: Optional[TemplateRepresentation] = None,
    ):
        self._clifford_1q_rep = clifford_1q_rep or Clifford1QZXZXZ()
        self._decomposer = Clifford2QToInterleavedCXLike(operation=operation)
        self._operation = operation
        super().__init__(Clifford2Q())

        # define in the constructor so that we only have one instance of each in memory
        self._sx = SXGate()

    @property
    def num_params_per_member(self):
        return (4 * self._clifford_1q_rep.num_params_per_member[0],) * 2

    def generate_template(self, qubits, parameters):
        # we need to be careful to add instructions in the same order as _decomposer, which yields
        # the trailing shape (num_decomposition_layers=4, num_qubits=2, 3)
        parameters = iter(parameters)
        sx = self._sx

        def sq_gates(qubit, qubit_parameters):
            yield CircuitInstruction(RZGate(next(qubit_parameters)), qubit)
            yield CircuitInstruction(sx, qubit)
            yield CircuitInstruction(RZGate(next(qubit_parameters)), qubit)
            yield CircuitInstruction(sx, qubit)
            yield CircuitInstruction(RZGate(next(qubit_parameters)), qubit)

        num_rz_layers = 4 * self._clifford_1q_rep.num_params_per_member[0]
        subsys_instructions_list = []
        for q0, q1 in zip(qubits[::2], qubits[1::2]):
            subsys_instructions_list.append(instrs := [])
            # the list on the following line is important: all of the qubit 0 parameters come first,
            # so we need to grab them all before next(parameters) will start getting the qubit 1
            # parameters
            q0, q1 = (q0,), (q1,)
            q0_parameters = iter([next(parameters) for _ in range(num_rz_layers)])

            instrs.append(grouping := [])
            grouping.extend(sq_gates(q0, q0_parameters))
            grouping.extend(sq_gates(q1, parameters))
            cx_like = CircuitInstruction(self._operation, q0 + q1)
            for _ in range(3):
                instrs.append([cx_like])
                instrs.append(grouping := [])
                grouping.extend(sq_gates(q0, q0_parameters))
                grouping.extend(sq_gates(q1, parameters))

        barrier = CircuitInstruction(Barrier(len(qubits)), qubits)
        for subsys_instructions in zip(*subsys_instructions_list):
            for grouping in subsys_instructions:
                yield from grouping
            yield barrier

    def parameter_values(self, members):
        return self._clifford_1q_rep.parameter_values(self._decomposer(members))
