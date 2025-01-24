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
Test maps/pauli.py
"""

import twirly as tw

from .. import TwirlyTestCase
from . import GroupMapTestCase, TensoredInjectionTestCase


class UniformPauliIntoClifford1QTestCase(TwirlyTestCase, TensoredInjectionTestCase):
    def setUp(self):
        self.group_maps = [tw.maps.UniformPauliIntoClifford1Q()]
        return super().setUp()


# TODO: Switch to TensoredInjectionTestCase when CliffordIntoUnitary.apply_reverse implemented
class UniformPauliIntoHaarUnitaryTestCase(TwirlyTestCase, GroupMapTestCase):
    def setUp(self):
        self.group_maps = [tw.maps.UniformPauliIntoHaarUnitary()]
        return super().setUp()


class UniformPauliIntoClifford2QTestCase(TwirlyTestCase, TensoredInjectionTestCase):
    def setUp(self):
        self.group_maps = [tw.maps.UniformPauliIntoClifford2Q()]
        return super().setUp()
