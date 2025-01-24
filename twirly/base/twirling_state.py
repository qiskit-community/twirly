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
Subsystems, indexed member arrays, and twirling contexts
"""

import pprint
import weakref
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple, Union

import numpy as np
from qiskit.circuit import CircuitInstruction, Qubit

from ..exceptions import TwirlingError
from ..utils import Shape, shape_tuple
from .twirling_group import MemberArray, TwirlingGroup

_f2 = lambda x: f"{x:.2f}"
_c2 = lambda x: f"{_f2(np.real(x))}{'+' if np.imag(x) >=0 else '-'}{_f2(np.abs(np.imag(x)))}"


def _format_array(array, max_width=50):
    ret = np.array2string(array, separator=",", formatter={"float_kind": _f2, "complex_kind": _c2})
    ret = ret.replace("\n", "").replace(" ", "")
    if len(ret) > max_width:
        ret = ret[: max_width - 4] + "...]"
    return ret


SubsystemsLike = Iterable[Iterable[Qubit]]


class Subsystems:
    __slots__ = ("num_qubits", "qubits")

    def __init__(self, qubits: SubsystemsLike, num_qubits: Optional[int] = None):
        r"""
        Args:
            qubits: An non-ragged iterable of iterables of qubits.
            num_qubits: The length of each sub-iterable of qubits.
                .. note::
                    This is necessary because we want to support length-0 :class:`~Subsystems`, and
                    we always want a well-defined value of ``num_subsystems``.
        """
        if isinstance(qubits, Subsystems):
            self.qubits: Tuple[Tuple[qubits]] = qubits.qubits
            self.num_qubits: int = qubits.num_qubits
        else:
            self.qubits: Tuple[Tuple[qubits]] = tuple(map(tuple, qubits))
            try:
                self.num_qubits: int = num_qubits or len(self.qubits[0])
            except IndexError as exc:
                raise TwirlingError("num_qubits must be provided when qubits is empty.") from exc
            if any(len(sys) != self.num_qubits for sys in self.qubits):
                raise TwirlingError("Subsystems cannot be ragged.")

    @property
    def num_subsystems(self) -> int:
        return len(self.qubits)

    @property
    def num_total_qubits(self) -> int:
        return self.num_qubits * self.num_subsystems

    @property
    def unordered_map(self) -> Dict[FrozenSet[Qubit], int]:
        return {frozenset(subsys): idx for idx, subsys in enumerate(self)}

    def __eq__(self, other):
        if self.num_subsystems != other.num_subsystems or self.num_qubits != other.num_qubits:
            return False
        return all(p == q for p, q in zip(self.flat_iter(), other.flat_iter()))

    def __iter__(self):
        # following the convention of NumPy, iterations are over the first axis
        yield from self.qubits

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.qubits[idx]
        return tuple(self.qubits[i] for i in idx)

    def __len__(self):
        return len(self.qubits)

    def __lt__(self, other):
        # TODO: this comparison is only used to get a consistent repr in other classes, and can
        # probably be removed after testing
        if self.num_qubits != other.num_qubits:
            return self.num_qubits < other.num_qubits
        if len(self) != len(other):
            return len(self) < len(other)
        for p, q in zip(self.flat_iter(), other.flat_iter()):
            if p != q:
                return str(p) < str(q)
        return False

    def __repr__(self):
        return f"Subsystems({self.qubits})"

    def difference(self, other: "Subsystems") -> "Subsystems":
        """Return all subsystems not present in the other subsystems, ignoring subsystem order."""
        other = other.unordered_map
        diff = (subsys for subsys in self if frozenset(subsys) not in other)
        return Subsystems(diff, num_qubits=self.num_qubits)

    def flat_iter(self):
        for subsystem in self.qubits:
            yield from subsystem

    @staticmethod
    def from_iterable(qubits, num_qubits):
        qubits = list(qubits)
        num_subsys = len(qubits) // num_qubits
        qubits = iter(qubits)
        return Subsystems(
            ((next(qubits) for _ in range(num_qubits)) for _ in range(num_subsys)),
            num_qubits,
        )

    def reshape(self, num_qubits: int) -> "Subsystems":
        if num_qubits == self.num_qubits:
            return self
        m = (self.num_qubits * self.num_subsystems) // num_qubits
        qubits = iter(self.flat_iter())
        return Subsystems(((next(qubits) for _ in range(num_qubits)) for _ in range(m)))


class IndexedMemberArray:
    r"""Helper class to act as members and reference targets in :class:`~TwirlingState`.

    Instances are immutable, and compare/hash via ``id``.
    """

    __slots__ = ("twirling_group", "qubits", "members", "__weakref__")

    def __init__(
        self,
        qubits: SubsystemsLike,
        twirling_group: TwirlingGroup,
        members: MemberArray,
    ):
        self.qubits = Subsystems(qubits, twirling_group.num_qubits)
        self.twirling_group = twirling_group
        self.members = members

        # TODO: think about eliminating checks
        assert members.shape[members.ndim - self.twirling_group.ndim :] == self.twirling_group.shape
        shape = self.twirling_group.member_array_shape(members)
        if shape:
            assert shape[0] == self.qubits.num_subsystems

    @property
    def full_shape(self):
        return self.twirling_group.member_array_shape(self.members)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def num_subsystems(self) -> int:
        return self.qubits.num_subsystems

    @property
    def num_qubits(self) -> int:
        return self.qubits.num_qubits

    @property
    def shape(self) -> Shape:
        return self.full_shape[1:]

    def __iter__(self):
        yield self.qubits
        yield self.twirling_group
        yield self.members

    def __lt__(self, other):
        # TODO: this is only used to get a consistent repr of TwirlingState, and can probably be
        # removed after prototyping
        if self.qubits != other.qubits:
            return self.qubits < other.qubits
        return False

    def __repr__(self):
        qubits = repr(self.qubits)
        members = _format_array(self.members)
        return (
            f"<IndexedMemberArray({qubits}, {self.twirling_group}, {members}) at {hex(id(self))}>"
        )

    def __str__(self):
        return f"({self.qubits}, {self.twirling_group}, {_format_array(self.members)})"

    @staticmethod
    def from_instructions(
        twirling_group: TwirlingGroup, instructions: Iterable[CircuitInstruction]
    ):
        all_qubits = set()
        subsystems = []
        members = []
        for instr in instructions:
            members.append(twirling_group.from_operation(instr.operation))
            if not all_qubits.isdisjoint(instr.qubits):
                raise ValueError(
                    f"Found overlapping instructions on {all_qubits.intersection(instr.qubits)}"
                )
            all_qubits.update(instr.qubits)
            subsystems.append(instr.qubits)
        members = np.array(members, dtype=twirling_group.dtype)
        return IndexedMemberArray(subsystems, twirling_group, members)

    def is_compatible(self, other: "IndexedMemberArray") -> bool:
        return (self.twirling_group == other.twirling_group) and (self.shape == other.shape)

    @staticmethod
    def join(triples: Iterable["IndexedMemberArray"]) -> "IndexedMemberArray":
        if len(triples) == 0:
            raise TwirlingError("At least one input is required.")
        for triple in triples[1:]:
            if not triple.is_compatible(triples[0]):
                raise TwirlingError(f"{triples[0]} is not compatible with {triple}.")

        qubits = (subsystem for triple in triples for subsystem in triple.qubits)
        members = np.concatenate([triple.members for triple in triples], axis=0)
        return IndexedMemberArray(qubits, triples[0].twirling_group, members)

    def pad_and_permute(self, qubits: Subsystems) -> "IndexedMemberArray":
        missing_qubits = qubits.difference(self.qubits)
        if len(missing_qubits) == 0:
            return self
        ids = self.twirling_group.id_member(shape_tuple(missing_qubits.num_subsystems, self.shape))
        padding = self.replace(members=ids, qubits=missing_qubits)
        return IndexedMemberArray.join([self, padding]).permute(qubits)

    def permute(self, qubits: Subsystems) -> "IndexedMemberArray":
        r"""Return a new :class:`IndexedMemberArray` with order matching ``qubits``.

        Both the member order (i.e. first index of ``qubits``) and the subsystem order (i.e. the
        second index of ``qubits``) are considered.

        Args:
            qubits: A 2D array of qubits, as in the constructor, containing exactly the subsystems
                present in this triple, but possibly with different member and subsystems orders.
        Return:
            This triple or a new triple.
        """
        if self.qubits == qubits:
            return IndexedMemberArray(*self)

        # first figure out which members are out of order, and change them
        subsys_map = self.qubits.unordered_map
        member_perm = [subsys_map[q] for q in map(frozenset, qubits) if q in subsys_map]
        members = self.members[member_perm, ...]

        # now permute each subsystem
        if self.num_qubits > 1:
            perms = []
            for idx, subsys in enumerate(qubits):
                perms.append([subsys.index(q) for q in self.qubits[member_perm[idx]]])
            perms = np.array(perms)
            perms = perms.reshape([-1] + [1] * self.ndim + [self.twirling_group.num_qubits])
            members = self.twirling_group.permute(members, perms)

        return IndexedMemberArray(qubits, self.twirling_group, members)

    def replace(
        self,
        qubits: Optional[SubsystemsLike] = None,
        twirling_group: Optional[TwirlingGroup] = None,
        members: Optional[MemberArray] = None,
    ):
        return IndexedMemberArray(
            self.qubits if qubits is None else qubits,
            self.twirling_group if twirling_group is None else twirling_group,
            self.members if members is None else members,
        )

    def split(self, qubits: Subsystems) -> Tuple["IndexedMemberArray", "IndexedMemberArray"]:
        r"""Return two new :class:`~IndexedMemberArray`\s that combine into this instance, up to
        order.

        The order of each subsystem is ignored. For example, if some row of ``qubits`` is ``[6,5]``,
        then this method will consider both ``[5,6]`` and ``[6,5]`` to be present, but will not
        perform any subsystem permutations; the subsystem orders already present in this instance
        will be maintained, see also :meth:`~permute`.

        Args:
            qubits: A 2D array of qubits, as in the constructor.
        Return:
            A pair ``(present, absent)``, where ``present`` contains all subsystems specified in
            ``qubits``, and ``absent`` contains the rest.
        """
        # find out which subsystems are present, irrespective of subsystem order
        subsys_map = self.qubits.unordered_map
        present = [subsys_map.pop(q) for q in map(frozenset, qubits) if q in subsys_map]
        absent = list(subsys_map.values())
        return self.subsys_slice(present), self.subsys_slice(absent)

    def subsys_slice(self, sl: Union[slice, List[int]]) -> "IndexedMemberArray":
        r"""Return a new :class:`~IndexedMemberArray` with sliced qubits and members."""
        return IndexedMemberArray(self.qubits[sl], self.twirling_group, self.members[sl, ...])


class TwirlingState:
    def __init__(
        self,
        shape: Shape = (),
        state: Optional[Iterable[IndexedMemberArray]] = None,
    ):
        self._triples = set()
        self._qubit_refs = {}
        self._shape = shape_tuple(shape)
        if state:
            for triple in state:
                self.add(triple)

    @property
    def shape(self) -> Shape:
        return self._shape

    def __iter__(self):
        yield from self._triples

    def __repr__(self):
        return f"TwirlingState({pprint.pformat(sorted(self._triples), width=100)})"

    def _add(self, triple: IndexedMemberArray, check_overlap: bool = True):
        # check_overlap exists because internal functions may want to overwrite existing values in
        # _qubit_ref
        if triple.shape != self._shape:
            raise TwirlingError(
                f"The triple ({triple}) has (qubit-discounted) member array shape {triple.shape} "
                f"instead of the expected shape {self._shape}."
            )
        for subsys_idx, subsys in enumerate(triple.qubits):
            for qubit_idx, qubit in enumerate(subsys):
                if check_overlap and qubit in self._qubit_refs:
                    raise TwirlingError(f"The structure repeats qubit {qubit} somewhere.")
                # TODO: this ref probably doesn't have to be weak if the non-const
                # methods of this class are written properly
                self._qubit_refs[qubit] = (weakref.ref(triple), subsys_idx, qubit_idx)
        self._triples.add(triple)

    def add(self, triple: IndexedMemberArray):
        # we're just taking away the option to check_overlap
        self._add(triple)

    def extract(self, qubits: Subsystems, group: TwirlingGroup) -> IndexedMemberArray:
        r"""Mutates this state by extracting group members for exactly the specified qubits.

        If ``qubits`` does not match the order (either member order or subsystem order), the members
        are permuted so that the output of this method matches the specification ``qubits`` exactly.
        If this state doesn't contain certain subsystems, identity members are created to fill the
        positions.

        Args:
            qubits: A 2D array of qubits, as in the constructor.
            group: The type of member to extract.
        Return:
            A group member array whose first index corresponds to the first index of ``qubits``.
        Raises:
            TwirlingError: If the wrong group type is found intersecting the provided qubits.
        """
        # loop through intersecting triples, collecting pieces of them that are a subset of qubits
        subsets = []
        missing_subsystems = set(map(frozenset, qubits))
        for triple in self.intersection(qubits):
            if triple.twirling_group != group:
                raise TwirlingError(
                    f"There is a group mismatch on the selected qubits {qubits}: {group} was "
                    f"specified, but {triple.twirling_group} was found on qubits {triple.qubits}."
                )
            overlap, extra = triple.split(qubits)
            subsets.append(overlap)
            missing_subsystems.difference_update(map(frozenset, overlap.qubits))

            # remove the triple, but add the extra bit back in (if non-empty)
            self._triples.remove(triple)
            for qubit in overlap.qubits.flat_iter():
                self._qubit_refs.pop(qubit)
            if extra.num_subsystems > 0:
                self._add(extra, check_overlap=False)

        # fill any missing subsystems with group identity members
        if missing_subsystems:
            missing_qubits = Subsystems(missing_subsystems)
            id_members = group.id_member(shape_tuple(missing_qubits.num_subsystems, self._shape))
            subsets.append(IndexedMemberArray(missing_qubits, group, id_members))

        # join what we have found together and force it into the order dictated by qubits
        return IndexedMemberArray.join(subsets).permute(qubits)

    def intersection(self, qubits: Subsystems) -> Iterable[IndexedMemberArray]:
        r"""Yield all triples that overlap with the given qubits."""
        qubits = set(qubits.flat_iter())
        while qubits:
            triple_ref = self._qubit_refs.get(qubits.pop(), [None])[0]
            if triple_ref is not None and triple_ref() is not None:
                qubits.difference_update(triple_ref().qubits.flat_iter())
                yield triple_ref()

    def pop(self, triple: IndexedMemberArray):
        self._triples.discard(triple)
