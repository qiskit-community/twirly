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
Group map implementations
"""

from .clifford_1q import Clifford1QIntoClifford2Q, Clifford1QIntoHaarUnitary
from .clifford_2q import Clifford2QToInterleavedCXLike
from .clifford import CliffordIntoUnitary, CliffordEmbedding, SmallCliffordGroupIntoUniformClifford
from .pauli import (
    UniformPauliIntoClifford1Q,
    UniformPauliIntoClifford2Q,
    UniformPauliIntoHaarUnitary,
)

from . import clifford_1q
from . import clifford_2q
from . import clifford
from . import pauli
