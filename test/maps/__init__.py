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
Test group map implementations
"""

from typing import Iterable
from unittest import SkipTest

import twirly as tw


class GroupMapTestCase:
    """Mixin for tests common to all group maps."""

    group_maps: Iterable[tw.GroupMap]

    def test_domain_shape_compat(self):
        """Test generic compatibility between apply(), domain_shape(), and codomain_shape()."""
        for group_map in self.group_maps:
            if group_map.num_codomains > 1 or group_map.num_domains > 1:
                raise SkipTest

            members = group_map.domain.sample((12, 3))
            output = group_map.apply(members)

            # check output shape is as predicted
            output_shape = group_map.codomain.member_array_shape(output)
            self.assertEqual(output_shape, group_map.codomain_shape((12, 3)))

            # check the round trip is as expected
            self.assertEqual(group_map.domain_shape(output_shape), (12, 3))


class HomomorphismTestCase(GroupMapTestCase):
    """Mixin for tests common to all homomorphisms."""

    group_maps: Iterable[tw.base.Homomorphism]

    def test_homomorphism_property(self):
        """Test that taking the product/inv in the domain and then applying the map is equivalent
        to taking the product/inv in the codomain.
        """
        for group_map in self.group_maps:
            domain_lhs, domain_rhs = group_map.domain.sample((2, 6, 3, 2))
            domain_dot = group_map.domain.dot(domain_lhs, domain_rhs)
            domain_inv = group_map.domain.inv(domain_lhs)

            codomain_lhs, codomain_rhs = (
                group_map.apply(domain_lhs),
                group_map.apply(domain_rhs),
            )
            codomain_dot = group_map.codomain.dot(codomain_lhs, codomain_rhs)
            codomain_inv = group_map.codomain.inv(codomain_lhs)

            self.assertTrue(group_map.codomain.equal(group_map.apply(domain_dot), codomain_dot))
            self.assertTrue(group_map.codomain.equal(group_map.apply(domain_inv), codomain_inv))


class InjectionTestCase(HomomorphismTestCase):
    """Mixin for tests common to all injections."""

    group_maps: Iterable[tw.base.Injection]

    def test_apply_round_trip(self):
        """Test round trip of apply() <--> apply_reverse()"""
        for group_map in self.group_maps:
            members = group_map.domain.sample((12, 3))
            output = group_map.apply(members)
            new_members = group_map.apply_reverse(output)
            self.assertTrue(group_map.domain.equal(members, new_members))


class TensoredInjectionTestCase(InjectionTestCase):
    """Mixin for tests common to all tensored injections."""

    group_maps: Iterable[tw.base.TensoredInjection]

    def test_codomain_shape_is_mult(self):
        """Test that the map modifies the shape by increasing the size of the first index."""
        for group_map in self.group_maps:
            ratio = group_map.num_qubits_ratio
            self.assertEqual(group_map.domain_shape((3, 4, 5)), (3 * ratio, 4, 5))
            self.assertEqual(group_map.codomain_shape((120, 4, 5)), (120 // ratio, 4, 5))
