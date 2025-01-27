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
Test maps/clifford_1q.py
"""

import twirly as tw

from .. import TwirlyTestCase
from . import GroupMapTestCase, TensoredInjectionTestCase


# TODO: Switch to TensoredInjectionTestCase when CliffordIntoUnitary.apply_reverse implemented
class Clifford1QIntoHaarUnitaryTestCase(TwirlyTestCase, GroupMapTestCase):
    def setUp(self):
        self.group_maps = [tw.maps.Clifford1QIntoHaarUnitary()]
        return super().setUp()


class Clifford1QIntoClifford2QTestCase(TwirlyTestCase, TensoredInjectionTestCase):
    def setUp(self):
        self.group_maps = [tw.maps.Clifford1QIntoClifford2Q()]
        return super().setUp()
