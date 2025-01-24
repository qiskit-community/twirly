# This code is part of Twirly.
#
# This is proprietary IBM software for internal use only, do not distribute outside of IBM
# Unauthorized copying of this file is strictly prohibited.
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
Group map implementations
"""

from . import clifford, clifford_1q, clifford_2q, pauli
from .clifford import (
    CliffordEmbedding,
    CliffordIntoUnitary,
    SmallCliffordGroupIntoUniformClifford,
)
from .clifford_1q import Clifford1QIntoClifford2Q, Clifford1QIntoHaarUnitary
from .clifford_2q import Clifford2QToInterleavedCXLike
from .pauli import (
    UniformPauliIntoClifford1Q,
    UniformPauliIntoClifford2Q,
    UniformPauliIntoHaarUnitary,
)
