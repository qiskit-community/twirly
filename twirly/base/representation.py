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
Twirling group circuit representation base classes
"""

import abc
from functools import cached_property
from typing import Iterable, Tuple, Union

import numpy as np
from qiskit.circuit import (
    CircuitInstruction,
    ParameterExpression,
    Qubit,
)

from .circuit_bundle import CircuitBundle
from .enumerations import EnumeratedTwirlingGroup
from .group_map import Identity, Injection
from .twirling_group import MemberArray, TwirlingGroup
from .twirling_state import Subsystems


class CircuitRepresentation(abc.ABC):
    def __init__(self, twirling_group: Union[TwirlingGroup, Injection]):
        r"""The ``twirling_group`` can naturally be a :class:`~TwirlingGroup`. However, it is also
        convenient to be able to use a :class:`~CircuitRepresentation` for other
        :class:`~TwirlingGroup`\s that are not explicitly accomodated by by the representation, but
        for which a natural injection exists into the native group assumed by the representation.
        For this reason, the ``domain`` can also be given as a :py:class:`Injection` whose domain
        members we wish to represent in a circuit form, and whose codomain is natural supported by
        the :class:`~CircuitRepresentation`.

        For example, one could write a generic :class:`~CircuitRepresentation` for
        :class:`~HaarUnitary` of some specific dimension, but then use the representation for some
        other :class:`~TwirlingGroup` with a natural, dimension-preserving, injection into
        :class:`~HaarUnitary`.
        """
        if isinstance(twirling_group, TwirlingGroup):
            twirling_group = Identity(twirling_group, twirling_group)
        self._injection = twirling_group

    @property
    def injection(self) -> Injection:
        return self._injection

    @property
    def twirling_group(self) -> TwirlingGroup:
        return self._injection.domain

    @abc.abstractmethod
    def append_instructions(
        self, circuit_bundle: CircuitBundle, qubits: Subsystems, members: MemberArray
    ):
        r"""Append circuit instructions to the circuit bundle that implement all of the ``members``
        on the provided ``qubits``.

        Args:
            circuit_bundle: The circuit bundle to append to.
            qubits: The qubits for the members to be implemented on. The number of qubits should be
                equal to the number of members times the number of qubits each member requires,
                as defined by the :py:class:`~TwirlingGroup` of this representation.
            members: A member array, whose first dimension is equal to the number of subsystems.
        """


class TemplateRepresentation(CircuitRepresentation):
    @property
    @abc.abstractmethod
    def num_params_per_member(self) -> Tuple[int]:
        r"""The number of :class:`qiskit.circuit.Parameter`\s used per group member for each
        respective qubit it acts on.

        For example, a value of ``(2, 4, 1)`` specifies that a single group member acts on three
        qubits (which typically coincides with :attr:`~TwirlingGroup.num_qubits` of the
        :attr:`~injection`\s codomain), where the first qubit requires two parameters, the second
        qubit requires four parameters, and the third qubit requires one parameters.
        """
        pass

    def append_instructions(
        self, circuit_bundle: CircuitBundle, qubits: Subsystems, members: MemberArray
    ):
        # the first dimension of members should coincide with the number of subsystems, so that the
        # first dimension of parameter_values(members) also coincides with the number of subsystems.
        # the product of the last dimensions of parameter_values(members) should correspond with the
        # number of parameters required for a single member, i.e. the sum of num_params_per_member.
        # any remaining product of dimensions must be the size of the circuit bundle.
        param_gen = circuit_bundle.parameter_generator
        for subsys, values in zip(qubits, self.parameter_values(members)):
            parameters = [
                param_gen.next(qubit)
                for n_params, qubit in zip(self.num_params_per_member, subsys)
                for _ in range(n_params)
            ]

            template = self.generate_template(subsys, parameters)
            values = values.reshape(-1, len(parameters)).T
            circuit_bundle.append_parametric(template, dict(zip(parameters, values)))

    @abc.abstractmethod
    def generate_template(
        self, qubits: Iterable[Qubit], parameters: Iterable[ParameterExpression]
    ) -> Iterable[CircuitInstruction]:
        r"""Return a collection of parameterized instructions capable of representing any single
        member acting on the given qubits.

        Args:
            qubits: The qubits to generate the template for a single member on.
            parameters: An iterable of parameters. This method will call ``next`` on an iterator of
                the parameters exactly ``sum(``\:attr:`~num_params_per_member`\``)`` times. All of
                the parameters for the first qubit should appear first, and then those for the
                second, and so on.
        Return:
            A sequence of circuit instructions acting as a template for any member of the group.
        """

    @abc.abstractmethod
    def parameter_values(self, members: MemberArray) -> np.ndarray:
        r"""Return a parameter array that can be applied to the template.

        The first axes of the returned array should correspond to the member array shape. For
        example, if ``members`` describes a ``(4, 3)`` array of group members, then the shape of
        the returned array should lead with ``(4, 3)``. The remaining dimensions of the returned
        array, when flattened, should correspond to the ``parameters`` accepted by
        :meth:`~generate_template`.

        Args:
            members: An array of members to compute parameter values for.
        Return:
            An array of parameter values, where each is a valid substitution for
            :meth:`generate_template`.
        """


class CachedTemplateRepresentation(TemplateRepresentation):
    _MAX_ELEMENTS = 20000

    def __init__(self, twirling_group: Union[TwirlingGroup, Injection]):
        super().__init__(twirling_group)

        if isinstance(self._injection.domain, EnumeratedTwirlingGroup):
            if self._injection.domain.num_members <= self._MAX_ELEMENTS:
                # create cache before patching, otherwise we will get infinite recursion
                self._all_parameter_values  # pylint: disable=pointless-statement
                self.parameter_values = self._parameter_values_from_cache

    @cached_property
    def _all_parameter_values(self):
        return self.parameter_values(self._injection.domain.all_members)

    def _parameter_values_from_cache(self, members):
        return self._all_parameter_values[members]
