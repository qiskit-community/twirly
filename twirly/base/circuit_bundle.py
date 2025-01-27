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
Circuit bundle
"""

from itertools import count, product
from typing import Dict, Iterable, List, Sequence

from qiskit.circuit import CircuitInstruction, Parameter, QuantumCircuit, Qubit
from qiskit_aer import AerSimulator

ParameterMap = Dict[Parameter, Sequence[float]]


class ParameterGenerator:
    r"""Generates parameters to be used in a circuit.

    Circuits order parameters by their names, alphabetically. The hardware probably wants parameter
    values separated by qubit. This implementation generates streams of parameters with a distinct
    prefix for each qubit, and such that temporal appearance of the parameter in the circuit matches
    alphabetical order for each qubit.
    """

    def __init__(self, qreg):
        self._qreg = qreg
        self._params = {}

    def iter(self, qubit: Qubit) -> Iterable[Parameter]:
        params = self._params.get(qubit, None)
        if params is None:
            if qubit not in self._qreg:
                raise ValueError(f"{qubit} is not part of {self._qreg}")
            prefix_template = f"{self._qreg.name}_{{:0{len(str(len(self._qreg) - 1))}d}}_"
            prefix = prefix_template.format(self._qreg.index(qubit))
            params = self._params[qubit] = map(Parameter, self._unique_names(prefix=prefix, n=2))

        return params

    def next(self, qubit: Qubit) -> Parameter:
        return next(self.iter(qubit))

    @staticmethod
    def _unique_names(*, prefix: str, n: int) -> Iterable[str]:
        """Generates an endless stream of unique names, in alphabetical order.

        Args:
            prefix: A fixed prefix to use in each name.
            n: Sets the expected upper-bound on the number of names to ``240**n``. A number too
                low will result in an excessive number of ``"z"``s following the prefix.
        Yield:
            Unique, alphabetically ordered names.
        """
        extra_iter = iter("z" * n for n in count())
        template = f"{prefix}{{}}{{}}{{:0{n}d}}"
        while True:
            extra = next(extra_iter)
            for base in product("abcdefghijklmnopqrstuvwxyz", repeat=n):
                for idx in range(10**n):
                    yield template.format(extra, "".join(base), idx)


class CircuitBundle:
    r"""
    An abstraction to hide the difference between a sequence of :math:`n`
    :class:`QuantumCircuit`\s, and a single parametric :class:`QuantumCircuit`
    with a list of :math:`n` parameter sets to bind to.
    """

    def __init__(self, base_circuit: QuantumCircuit):
        # _params is a dict from Parameters to a list of values equal to the length of the bundle
        # _circuits either has one entry, so that the bundle size is exclusively controlled by
        # _params, or it has the same length as the lists in _params, where parameters are
        # distributed 1-1
        self._circuits = [base_circuit]
        self._metadata = None
        self._params = {}
        # start with size None instead of 1 to allow correct behaviour in the
        # edge-case of a length-1 bundle
        self._size = None

        self._param_gen = ParameterGenerator(base_circuit.qregs[0])

    @property
    def base_circuit(self) -> QuantumCircuit:
        if not self.has_single_base:
            raise RuntimeError("This bundle does not have a single base circuit.")
        return self._circuits[0]

    @property
    def has_single_base(self) -> bool:
        return len(self._circuits) == 1

    @property
    def metadata(self) -> List[Dict]:
        if self._size is None:
            return [{}]
        elif self._metadata is None:
            self._metadata = [{} for _ in range(self._size)]
        return self._metadata

    @property
    def parameter_generator(self) -> ParameterGenerator:
        return self._param_gen

    def __getitem__(self, idx):
        circuit = self._circuits[0 if len(self._circuits) < len(self) else idx]
        params = {param: values[idx] for param, values in self._params.items()}
        return circuit.assign_parameters(params)

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __len__(self):
        return self._size or 1

    def append_multiple_static(self, instructions: Sequence[CircuitInstruction]):
        r"""Appends one non-parametric instruction to each circuit in the bundle.

        Calling this method will result in a bundle that, internally, stores multiple circuits, in
        contrast to :meth:`append_parametric`, which adds to parameter value lists.

        Args:
            instructions: One non-parametric circuit instruction for each circuit in the bundle.
        Raises:
            ValueError: If any of the instructions are parametric, or if they do not match the
                length of this bundle.
        """
        n = len(instructions)
        if self._size is None:
            # this is the first time either append*() method has been called
            self._circuits.extend(self._circuits[0].copy() for _ in range(n - 1))
            self._size = n
        elif self._size != n:
            raise ValueError(f"Expected {self._size} instructions instead of {n}.")
        elif len(self._circuits) == 1 and self._size > 1:
            # only _append_parametric() has been called before this
            self._circuits.extend(self._circuits[0].copy() for _ in range(n - 1))

        for circuit, instruction in zip(self._circuits, instructions):
            if instruction.operation.is_parameterized():
                raise ValueError("Cannot use parametric instructions with append_static().")
            circuit._append(instruction)

    def append_parametric(
        self, instructions: Iterable[CircuitInstruction], parameters: ParameterMap
    ):
        r"""Appends the same parametric instruction to each circuit in the bundle, with
        corresponding parameter values.

        Args:
            instructions: Circuit instructions to add to each circuit in the bundle.
            parameters: A mapping from parameters in the instructions to a sequence of values.
        Raises:
            ValueError: If some list of values does not correspond to the length of this bundle.
        """
        for param, values in parameters.items():
            if self._size is None:
                self._size = len(values)
            elif len(values) != self._size:
                raise ValueError(f"{param}: {values} should have length {self._size}")
            self._params[param] = values

        for instruction in instructions:
            for circuit in self._circuits:
                circuit._append(instruction)

    def simulate_counts(self):
        r"""Yield ideal simulation results of the circuits in this bundle."""
        backend = AerSimulator()
        for circuit in self:
            yield backend.run(circuit).result().get_counts()

    def update_metadata(self, metadata_list: Sequence[Dict]):
        n = len(metadata_list)
        if self._size is None:
            self._size = n
        elif self._size != n:
            raise ValueError(f"Expected {self._size} metadata updates instead of {n}")
        if self._metadata is None:
            self._metadata = [{} for _ in range(n)]
        for old, new in zip(self._metadata, metadata_list):
            old.update(new)
