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
Test utils.py
"""

import numpy as np

import twirly.utils as tu

from . import TwirlyTestCase


class TestUtil(TwirlyTestCase):
    def test_iter_along_axis(self):
        arr = np.random.default_rng().random((2, 3, 4))

        # axis 0
        for sl0, sl1 in zip(tu.iter_along_axis(arr, 0), [arr[0, ...], arr[1, ...]]):
            self.assertTrue(np.allclose(sl0, sl1))

        # axis 1
        for sl0, sl1 in zip(tu.iter_along_axis(arr, 1), [arr[:, 0, :], arr[:, 1, :], arr[:, 2, :]]):
            self.assertTrue(np.allclose(sl0, sl1))

        # axis 2
        for sl0, sl1 in zip(tu.iter_along_axis(arr, 2), [arr[..., 0], arr[..., 1], arr[..., 2]]):
            self.assertTrue(np.allclose(sl0, sl1))
