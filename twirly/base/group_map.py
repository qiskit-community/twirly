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
Group map and homomorphism base classes
"""

from typing import Iterable, Union

import abc
import inspect
from functools import cached_property

from ..exceptions import TwirlingError
from .enumerations import EnumeratedTwirlingGroup
from .twirling_group import TwirlingGroup

TwilingGroupArg = Union[Iterable[TwirlingGroup], TwirlingGroup]


class GroupMap(metaclass=abc.ABCMeta):
    r"""A function mapping group member arrays from domains to codomains.

    This class is puposefully more general than group homomorphisms, i.e. functions :math:`\phi:
    G\rightarrow H` for two groups :math:`G,H` such that :math:`\phi(a\circ b) =
    \phi(a)\circ\phi(b)`. For example, because multiple domains are allowed, a group map can define
    conjugation. Suppose :math:`H` is a normal subgroup of :math:`G` via the injection
    :math:`\phi:H\rightarrayw G`. Then an instance of this class might represent the conjugation
    :math:`g:(H,G)\rightarrow H, (h,g)\mapsto \phi^{-1}(g\circ \phi(h)\circ g^{-1})`, which is not a
    homomorphism when interpreting the domain to be the direct product :math:`H\times G`.
    """

    def __init__(self, domains: TwilingGroupArg, codomains: TwilingGroupArg):
        self._domains = (domains,) if isinstance(domains, TwirlingGroup) else tuple(domains)
        self._codomains = (codomains,) if isinstance(codomains, TwirlingGroup) else tuple(codomains)
        self._check_restrictions(self._domains, self._codomains)

    @property
    def codomain(self):
        if self.num_codomains > 1:
            raise NotImplementedError(
                "This GroupMap contains multiple codomains, use the 'codomains' attr instead."
            )
        return self.codomains[0]

    @property
    def codomains(self):
        return self._codomains

    @property
    def domain(self):
        if self.num_domains > 1:
            raise NotImplementedError(
                "This GroupMap contains multiple domains, use the 'domains' attr instead."
            )
        return self.domains[0]

    @property
    def domains(self):
        return self._domains

    @property
    def num_codomains(self):
        return len(self._codomains)

    @property
    def num_domains(self):
        return len(self._domains)

    @classmethod
    def _check_restrictions(cls, domains, codomains):
        if len(domains) == 0:
            raise ValueError("At least one domain is required.")
        if len(codomains) == 0:
            raise ValueError("At least one codomain is required.")

    @abc.abstractmethod
    def apply(self, *input_members, **qualifiers):
        r"""Apply this map to return members from the codomains.

        Args:
            input_members: Valid member arrays for each of the :attr:`domains`, respectively.
            qualifiers: Additional information that the map may need.
        Return:
            A tuple of member arrays for the respective :attr:`codomains`. However, if there is a
            single codomain, i.e. :attr:`~num_codomains` is one, instead return the codomain member
            array itself, rather than a length-1 tuple. This convention is assumed by
            :class:`~GroupMapComposition`.
        """

    def codomain_shape(self, input_shape, **qualifiers):
        if self.num_codomains > 1:
            raise NotImplementedError(
                "This GroupMap contains multiple codomains, use the codomain_shapes method instead."
            )
        return self.codomain_shapes(input_shape, **qualifiers)[0]

    @abc.abstractmethod
    def codomain_shapes(self, *input_shapes, **qualifiers):
        r"""Return the shapes expected given input member shapes for each domain."""

    def domain_shape(self, output_shape, **qualifiers):
        if self.num_domains > 1:
            raise NotImplementedError(
                "This GroupMap contains multiple domains, use the domains_shapes method instead."
            )
        return self.domain_shapes(output_shape, **qualifiers)[0]

    @abc.abstractmethod
    def domain_shapes(self, *output_shapes, **qualifiers):
        r"""Return the shapes expected given output member shapes for each codomain."""

    def _apply_strict(self, input_members, **qualifiers):
        ret = self.apply(*input_members, **qualifiers)
        return (ret,) if self.num_codomains == 1 else ret

    def __call__(self, *input_members, **qualifiers):
        return self.apply(*input_members, **qualifiers)

    def __matmul__(self, other):
        cls = GroupMapComposition
        maps = (self, other)
        if all(isinstance(m, TensoredInjection) for m in maps):
            cls = InjectionComposition
        elif all(isinstance(m, Injection) for m in maps):
            cls = InjectionComposition
        elif all(isinstance(m, Homomorphism) for m in maps):
            cls = HomomorphismComposition
        return cls(self, other)

    def __repr__(self):
        domains = self._domains[0] if self.num_domains == 1 else self._domains
        codomains = self._codomains[0] if self.num_codomains == 1 else self._codomains
        return f"{type(self).__name__}({domains}, {codomains})"


class GroupMapComposition(GroupMap):
    def __init__(self, *maps):
        if len(maps) == 0:
            raise TwirlingError("At least one GroupMap is required.")
        self._maps = []
        prev = None
        for item in maps:
            # the double loop is to flatten nested GroupMapCompositions via associativity
            for cur in item if isinstance(item, type(self)) else [item]:
                self._append(cur, prev)
                prev = cur

        super().__init__(self._maps[-1][0].domains, self._maps[0][0].codomains)

    def _append(self, cur, prev):
        if not isinstance(cur, GroupMap):
            raise TwirlingError(f"Expected {cur} to be a GroupMap.")
        if prev is not None and not prev.domains == cur.codomains:
            raise TwirlingError(f"Cannot compose {prev} with {cur}. Maybe you reversed the order?")
        cur_qualifer_names = set()
        for idx, parameter in enumerate(inspect.signature(cur.apply).parameters.values()):
            if parameter.kind == parameter.VAR_KEYWORD:
                raise TwirlingError(
                    "Cannot compose GroupMaps with generic **kwargs because distributing "
                    "qualifiers may be ambiguous."
                )
            elif parameter.kind == parameter.KEYWORD_ONLY:
                cur_qualifer_names.add(parameter.name)
            elif idx >= cur.num_domains and parameter.kind == parameter.POSITIONAL_OR_KEYWORD:
                cur_qualifer_names.add(parameter.name)
        self._maps.append((cur, cur_qualifer_names))

    def apply(self, *input_members, **qualifiers):
        ret = input_members
        for group_map, names in reversed(self._maps):
            kwargs = {name: qualifiers[name] for name in names.intersection(qualifiers)}
            ret = group_map._apply_strict(ret, **kwargs)
        return ret if self.num_codomains > 1 else ret[0]

    def codomain_shapes(self, *input_shapes, **qualifiers):
        ret = input_shapes
        for group_map, names in reversed(self._maps):
            kwargs = {name: qualifiers[name] for name in names.intersection(qualifiers)}
            ret = group_map.codomain_shapes(*ret, **kwargs)
        return ret

    def domain_shapes(self, *output_shapes, **qualifiers):
        ret = output_shapes
        for group_map, names in self._maps:
            kwargs = {name: qualifiers[name] for name in names.intersection(qualifiers)}
            ret = group_map.domain_shapes(*ret, **kwargs)
        return ret

    def __iter__(self):
        for h, _ in self._maps:
            yield h

    def __repr__(self):
        return " @ ".join(map(repr, self))


class Homomorphism(GroupMap):
    """A :class:`~GroupMap` where group multiplication in the domain is consistent with
    multiplicaton in the codomain.
    """

    def codomain_dot(self, lhs, rhs, lhs_in_domain=True, rhs_in_domain=True, **qualifiers):
        if lhs_in_domain:
            lhs = self._apply_strict(lhs, **qualifiers)
        if rhs_in_domain:
            rhs = self._apply_strict(rhs, **qualifiers)
        ret = tuple(
            group_map.dot(lhs_item, rhs_item)
            for group_map, lhs_item, rhs_item in zip(self.codomains, lhs, rhs)
        )
        return ret if self.num_codomains > 1 else ret[0]


class Injection(Homomorphism):
    r"""A :class:`~Homomorphism` that is one-to-one.

    That is, every element of the domain is mapped to a unique element of the codomain.
    """

    @abc.abstractmethod
    def apply_reverse(self, *output_members, **qualifiers):
        r"""Map from the image of the domain back into the domain.

        This method can always be implemented because the image of an injection is isomorphic
        to the domain.

        Raises:
            TwirlingError: If any of the output members are not in the image of the domain.
        """


class TensoredInjection(Homomorphism):
    r"""An :class:`~Injection` that maps :math:`m` elements of the domain to a single element of the
    codomain by taking their tensor product.

    A convention is enforced where the factors are pulled out of the first axis of a group member
    array in the domain. For example, if a group member array a domain with a two-qubit group has
    shape ``(9,2,4)``, and the codomain is a six-qubit group, then the output group member array
    will have shape ``(3,2,4)`` because :math:`m=3` in this example, so we need to use three members
    of the domain to get one member in the codomain.
    """

    def __init__(self, domains: TwilingGroupArg, codomains: TwilingGroupArg):
        super().__init__(domains, codomains)
        if self.codomain.num_qubits % self.domain.num_qubits != 0:
            raise ValueError("The domain size does not evenly divide the codomain size.")

    @property
    def num_qubits_ratio(self):
        return self.codomain.num_qubits // self.domain.num_qubits


class HomomorphismComposition(GroupMapComposition, Homomorphism):
    pass


class InjectionComposition(HomomorphismComposition, Injection):
    def apply_reverse(self, *output_members, **qualifiers):
        ret = output_members
        for group_map, names in self._maps:
            kwargs = {name: qualifiers[name] for name in names.intersection(qualifiers)}
            ret = group_map.apply_reverse(*ret, **kwargs)
            ret = (ret,) if group_map.num_domains == 1 else ret
        return ret if self.num_codomains > 1 else ret[0]


class TensoredInjectionComposition(InjectionComposition, TensoredInjection):
    pass


class Conjugation(GroupMap):
    def __init__(self, normal_injection):
        normal_subgroup = normal_injection.domain
        parent = normal_injection.codomain
        self._injection = normal_injection
        super().__init__((normal_subgroup, parent), normal_subgroup)

    def apply(self, subgroup_members, parent_members):
        subgroup_members = self._injection.apply(subgroup_members)
        parent = self._injection.codomain
        conj = parent.dot(parent_members, parent.dot(subgroup_members, parent.inv(parent_members)))
        return self._injection.apply_reverse(conj)

    def codomain_shapes(self, *input_shapes):
        return input_shapes

    def domain_shapes(self, *output_shapes):
        return output_shapes


class Identity(TensoredInjection):
    @classmethod
    def _check_restrictions(cls, domains, codomains):
        super()._check_restrictions(domains, codomains)
        if domains != codomains:
            raise ValueError(f"The domains {domains} and codomains {codomains} must be equal.")

    def apply(self, *domain_members):
        return domain_members if self.num_domains > 1 else domain_members[0]

    def apply_reverse(self, *codomain_members):
        return codomain_members if self.num_domains > 1 else codomain_members[0]

    def codomain_shapes(self, *input_shapes):
        return input_shapes

    def domain_shapes(self, *output_shapes):
        return output_shapes


class CachedMapMixin:
    _MAX_ELEMENTS = 20000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.num_domains > 1:
            raise TwirlingError("CachedMapMixin not supported for multiple domains.")

        if isinstance(self.domain, EnumeratedTwirlingGroup):
            if self.domain.num_members <= self._MAX_ELEMENTS:
                # create cache before patching, otherwise we will get infinite recursion
                self._all_codomain_members  # pylint: disable=pointless-statement
                self.apply = self._apply_from_cache

    @cached_property
    def _all_codomain_members(self):
        return self.apply(self.domain.all_members)

    def _apply_from_cache(self, members):
        return self._all_codomain_members[members]

    def __repr__(self):
        return f"{type(self).__name__}({self.domain}, {self.codomain})"
