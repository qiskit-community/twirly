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
Test representations/clifford.py
"""

import twirly as tw

from .. import TwirlyTestCase
from . import TemplateRepresentationTestCase


class TestClifford1QZXZXZ(TwirlyTestCase, TemplateRepresentationTestCase):
    def setUp(self) -> None:
        self.num_members = 3
        self.depth = 4
        self.num_randomizations = 5
        self.rep = tw.representations.Clifford1QZXZXZ()
        super().setUp()


class TestClifford2QCXRepresentation(TwirlyTestCase, TemplateRepresentationTestCase):
    def setUp(self) -> None:
        self.num_members = 2
        self.depth = 4
        self.num_randomizations = 3
        self.rep = tw.representations.Clifford2QCXLikeRepresentation()
        super().setUp()
