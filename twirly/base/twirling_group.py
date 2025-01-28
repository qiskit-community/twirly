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
Twirling group base class
"""

import abc
from functools import reduce
from typing import Mapping

import numpy as np
from qiskit.circuit.operation import Operation

from ..exceptions import TwirlingError
from ..utils import Shape, shape_tuple

MemberArray = np.ndarray


class TwirlingGroup(abc.ABC):
    r"""
    Abstract base class for a group :math:`G` with a unitary representation on :math:`n`\-qubits.

    Each group member :math:`g\in G` is specified by a ``numpy.ndarray`` of a subclass-dependent
    ``dtype``. For example, members of full unitary group on n-qubits can simply be specified in
    terms of their :math:`2^n\times 2^n` complex matrix elements. However, this is a special case,
    where the unitary representation happens to correspond with the group member specification.
    Instead, some other specification is typically used that depends on the structure of the group
    :math:`G`. For example, the Clifford group members can be specified in terms of their augmented
    binary symplectic matrices. Or, a small finite group can simply be enumerated by integers, in
    which case the ``numpy.ndarray`` specifications of individual elements would be 0-dimensional
    (i.e. have the shape tuple ``()``).

    Implementations of this abstraction define how to do group operations (notably, :meth:`dot` and
    :meth:`inv`) given specifications of members.

    All relevant class methods accept arrays that are vectorized over members. For instance, suppose
    that for some group an individual member has a specification with shape ``(2, 4)`. Then an array
    of shape ``(5,6,2,4)``, assuming it has valid entries, specifies an array of shape ``(5,6)`` of
    group members. The properties :attr:`shape`, :attr:`ndim`, and :attr:`size` are in reference to
    the specification of a single member. Any two member arrays of the same shape can be multiplied
    in a vectorized fashion.
    """

    def __init__(self, num_qubits: int):
        self._num_qubits = num_qubits

    @property
    def dim(self) -> int:
        r"""The Hilbert space dimension of an individual group member's unitary representation."""
        return 2**self.num_qubits

    @property
    @abc.abstractmethod
    def dtype(self) -> type:
        r"""The ``numpy.ndarray`` datatype of group member specifications."""

    @property
    def ndim(self) -> int:
        r"""The length of :attr:`shape`."""
        return len(self.shape)

    @property
    def num_qubits(self) -> int:
        r"""The number of qubits each member of the group acts on."""
        return self._num_qubits

    @property
    def size(self) -> int:
        r"""The product of the values of :attr:`shape`."""
        return int(np.prod(self.shape, dtype=int))

    @property
    @abc.abstractmethod
    def shape(self) -> Shape:
        r"""The shape of the specification of a single group member."""

    def empty(self, shape: Shape = ()) -> MemberArray:
        r"""An empty member array of the given member array shape."""
        return np.empty(shape_tuple(shape, self.shape), dtype=self.dtype)

    def equal(self, lhs: MemberArray, rhs: MemberArray) -> bool:
        r"""Return whether all members of ``lhs`` and ``rhs`` are equal, with broadcasting."""
        return np.allclose(lhs, rhs)

    @abc.abstractmethod
    def id_member(self, shape: Shape = ()) -> MemberArray:
        r"""The identity element of the group, replicated into the given shape."""

    @abc.abstractmethod
    def inv(self, members: MemberArray) -> MemberArray:
        r"""The group inverse of each member in the give member array."""

    @abc.abstractmethod
    def dot(self, lhs: MemberArray, rhs: MemberArray) -> MemberArray:
        r"""The group product of each pair of members in the member arrays.

        Implementations should support broadcasting of singleton dimensions.
        """

    def dot_along_axis(
        self, members: MemberArray, axis: int = 0, reverse_order: bool = True
    ) -> MemberArray:
        shape = self.member_array_shape(members)
        ndim = len(shape)
        if axis < 0:
            axis = ndim + axis
        if 0 < axis >= ndim:
            raise TwirlingError(f"axis={axis} is out of bounds for member array shape {shape}.")

        def get_slice(idx):
            sl = slice(None, None, None)
            idx += reverse_order * (shape[axis] - 2 * idx - 1)
            return members[tuple(idx if jdx == axis else sl for jdx in range(ndim))]

        return reduce(
            self.dot,
            map(get_slice, range(shape[axis])),
            self.id_member((1,) * (ndim - 1)),
        )

    def find_orbit(self, member: MemberArray, max_order: int = 1000) -> MemberArray:
        r"""Return a 1-D member array consisting of the powers of the provided ``member``.

        Args:
            member: The member to compute the order for.
            max_order: When to give up finding the orbit.

        Return:
            The orbit of ``member``, starting with :attr:`id_member`.

        Raises:
            TwirlingError: If the orbit is found to be larger than ``max_order``.
        """
        orbit = [self.id_member()]
        for _ in range(max_order):
            orbit.append(self.dot(orbit[-1], member))
            if self.equal(orbit[0], orbit[-1]):
                return np.array(orbit[:-1], dtype=self.dtype)
        raise TwirlingError(f"The order of {member} exceeds max_order={max_order}.")

    def find_orders(self, members: MemberArray, max_order: int = 1000) -> Mapping[int, MemberArray]:
        r"""Separate the members by order.

        Args:
            members: A member array.
            max_order: When to give up finding the orbit for each member.

        Return:
            A dictionary mapping order values to a (flat) member array where each member has
            that order.
        """
        orders = {}
        for member in self.ravel(members):
            order = self.find_orbit(member, max_order).shape[0]
            if order not in orders:
                orders[order] = []
            orders[order].append(member)
        return {order: np.array(m, dtype=self.dtype) for order, m in orders.items()}

    def from_operation(self, operation: Operation) -> MemberArray:
        r"""
        Attempt to convert an operation into a 0D group member array.

        Args:
            operation: An operation to convert.
        Return:
            A 0D group member array, i.e. an array with shape :attr:`~shape`.
        Raises:
            TwirlingError: If the operation cannot be converted, or if the operation has a size
            incompatible with :attr:`~num_qubits`.
        """
        raise TwirlingError(f"{self} is unable to convert the operation {operation}.")

    def member_array_shape(self, members: MemberArray) -> Shape:
        r"""The shape of members discounting the shape of each member, :attr:`shape`."""
        return members.shape[: members.ndim - self.ndim]

    def member_array_size(self, members: MemberArray) -> int:
        return int(np.prod(self.member_array_shape(members), dtype=int))

    def outer(self, *members: MemberArray, ravel: bool = False) -> MemberArray:
        r"""Take the outer product over all given member arrays.

        This method differs from :meth:`~dot` in that all possible products are returned, rather
        than products zipped over two member arrays. The returned member array shape is the
        concatenation of each member array shape.

        For example, when given two 1-D member arrays, the following should be ``True``:

        .. code-block::
            np.allclose(group.outer(a, b), group.dot(a[:, np.newaxis, ...], a[np.newaxis, :, ...]))

        Args:
            members: Member arrays to take the outer product of.
            ravel: Whether to return as a 1D arary of members.
        Returns:
            All possible products of input member arrays.
        """
        if len(members) == 1:
            return members[0]
        shapes = list(map(self.member_array_shape, members))
        ret = self.empty(shapes)
        sl = slice(None, None, None)
        broadcasted = []
        start = stop = 0
        last = sum(map(len, shapes))
        for member in members:
            stop += member.ndim - self.ndim
            args = tuple(sl if start <= i < stop or i >= last else None for i in range(ret.ndim))
            start = stop
            broadcasted.append(member[args])
        products = reduce(self.dot, broadcasted)
        return self.ravel(products) if ravel else products

    @abc.abstractmethod
    def permute(self, members: MemberArray, permutations: np.ndarray) -> MemberArray:
        r"""Performs a subsystem permutation of each member.

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

    def ravel(self, members: MemberArray) -> MemberArray:
        r"""Returns a flattened view of the given members array.

        This method differs from ``numpy.ravel`` in that it does not ravel the dimensions associated
        with :attr:`~shape`. The returned view always has ``ndim`` equal to one more than
        :attr:`ndim`, even if the provided ``members`` has no axes.
        """
        return members.reshape(members.size // self.size, *self.shape)

    def reshape(self, members: MemberArray, *shape: Shape) -> MemberArray:
        r"""Returns a reshaped view of the given members array."""
        return members.reshape(shape_tuple(shape, self.shape))

    @abc.abstractmethod
    def sample(self, shape: Shape, seed=None) -> MemberArray:
        r"""Randomly sampled members of the group according to some distribution."""

    def __eq__(self, other: "TwirlingGroup") -> bool:
        return type(self) is type(other) and self.num_qubits == other.num_qubits

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.num_qubits})"
