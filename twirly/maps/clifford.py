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
Clifford group maps
"""

from typing import Optional

import numpy as np

from ..base import GroupMap, TensoredInjection
from ..exceptions import TwirlingError
from ..groups import HaarUnitary, SmallCliffordGroup, UniformClifford
from ..utils import iter_as_shape, shape_tuple


class CliffordIntoUnitary(TensoredInjection):
    @classmethod
    def _check_restrictions(cls, domains, codomains):
        domain, codomain = domains[0], codomains[0]
        if type(domain) is not UniformClifford:
            raise TwirlingError(f"The domain {domain} must be UniformClifford.")
        if type(codomain) is not HaarUnitary:
            raise TwirlingError(f"The domain {codomain} must be HaarUnitary.")
        if domain.num_qubits != codomain.num_qubits:
            raise TwirlingError(
                "CliffordIntoUnitary cannot change the number of qubits. Consider using this map "
                "in conjunction with CliffordEmbedding."
            )

    def apply(self, members):
        ret = self.codomain.empty(members.shape[:-2])
        for cliff, unitary in zip(self.domain.to_cliffords(members), self.codomain.ravel(ret)):
            unitary[:, :] = cliff.to_operator()
        return ret

    def apply_reverse(self, members):
        raise NotImplementedError()

    def codomain_shapes(self, *input_shapes):
        return input_shapes

    def domain_shapes(self, *output_shapes):
        return output_shapes


class SmallCliffordGroupIntoUniformClifford(TensoredInjection):
    @classmethod
    def _check_restrictions(cls, domains, codomains):
        domain, codomain = domains[0], codomains[0]
        if not isinstance(domain, SmallCliffordGroup):
            raise TwirlingError(f"The domain {domain} must be a subclass of SmallCliffordGroup.")
        if type(codomain) is not UniformClifford:
            raise TwirlingError(f"The domain {codomain} must be UniformClifford.")
        if domain.num_qubits != codomain.num_qubits:
            raise TwirlingError(
                "SmallCliffordGroupIntoUniformClifford cannot change the number of qubits. "
                "Consider using this map in conjunction with CliffordEmbedding."
            )

    def apply(self, domain_members):
        return self.domain._all_parent_members[domain_members]

    def apply_reverse(self, codomain_members):
        return self.domain._parent_lookup(codomain_members)

    def codomain_shapes(self, *input_shapes):
        return input_shapes

    def domain_shapes(self, *output_shapes):
        return output_shapes


def divide(x, y):
    z = x // y
    return z if y * z >= x else z + 1


class CliffordEmbedding(GroupMap):
    def apply(
        self,
        members: np.ndarray,
        idxs: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        output_shape: Optional[tuple] = None,
    ) -> np.ndarray:
        r"""
        Args:
            members: An array of members belonging to the domain of shape
                ``(s1,s2,...,sk,2*m,2*m+1)``.
            idxs: An array of shape ``(s1,s2,...,sk,k)`` specifying which output member to embed
                into. The last index of dimension ``k`` is a vector referencing indices of the
                output array of members.
            positions: An array of shape ``(s1,s2,...,sk,m)`` defining how to embed each subsystem
                of each Clifford. The last index of dimension ``m`` is a vector referencing
                subsystem indices.
            output_shape: The shape of the output array of members belonging to the codomain.
        Return:
            The embedded members.
        """
        # TODO: fix
        m = self.domain.num_qubits
        n = self.codomain.num_qubits
        input_shape = members.shape[: members.ndim - self.domain.ndim]
        inputs_per_output = n // m  # only used for automatic argument setting

        # handle the edge case of single shapeless member as follows
        compress_singleton = False
        if input_shape == ():
            input_shape = (1,)
            compress_singleton = True

        # handle missing output_shape or idxs
        if output_shape is None and idxs is None:
            # default to packing as densly as possible
            d = input_shape[-1]
            output_shape = shape_tuple(input_shape[:-1], divide(d, inputs_per_output))
            idxs = np.moveaxis(np.indices(input_shape), 0, -1)
            idxs[..., -1] = np.arange(output_shape[-1]).repeat(inputs_per_output)[:d]
        elif output_shape is None:
            output_shape = tuple(max(dim) + 1 for dim in idxs)

        # handle missing positions
        if positions is None:
            positions = np.tile(
                np.arange(m * inputs_per_output),
                shape_tuple(input_shape[:-1], output_shape[-1]),
            )[..., : input_shape[-1] * m].reshape(shape_tuple(input_shape, m))
        elif positions.ndim < len(input_shape) + 1:
            positions = positions.reshape(
                (1,) * (len(input_shape) + 1 - positions.ndim) + positions.shape
            )

        positions = iter_as_shape(positions, input_shape)
        idxs = iter_as_shape(idxs, input_shape)

        ret = self.codomain.id_member(output_shape)
        flat_ret = ret.reshape(output_shape + (-1,))
        fancy_slice = np.empty((2 * m, 2 * m + 1), dtype=int)
        for idx, member, position in zip(idxs, self.domain.ravel(members), positions):
            fancy_slice[:m, :m] = (2 * n + 1) * (position[0] + np.arange(m)[:, None]) + position[
                None, :
            ]
            fancy_slice[m:, :m] = fancy_slice[:m, :m] + n * (2 * n + 1)
            fancy_slice[:, m : 2 * m] = fancy_slice[:, :m] + n
            fancy_slice[:, -1] = fancy_slice[:, 0] + (
                (2 * n + 1) * position[0] - fancy_slice[0, 0] + 2 * n
            )
            flat_ret[tuple(idx) + (fancy_slice,)] = member

        if compress_singleton:
            ret = ret.reshape(2 * n, 2 * n + 1)

        return ret

    def codomain_shapes(self, *input_shapes):
        # TODO
        return "something complicated"

    def domain_shapes(self, *output_shapes):
        # TODO
        return "something complicated"
