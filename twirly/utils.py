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
Utility functions and classes
"""

from typing import Iterable, Tuple, Union

from _thread import RLock
from collections.abc import Iterable as IterableBase
from itertools import product

import numpy as np
from numpy.random import default_rng, Generator
from numpy.typing import DTypeLike

Shape = Tuple[int, ...]
ShapeInput = Union[int, Iterable["ShapeInput"]]
Rng = Union[None, Generator, int]

DEFAULT_RNG = default_rng()


def flatten_to_ints(arg: ShapeInput) -> Iterable[int]:
    """
    Yield one integer at a time.

    Args:
        arg: The integer or iterable of integers to yield recursively.

    Yields:
        The flattened integers.

    Raises:
        ValueError: If an input is not an iterable or an integer.
    """
    for item in arg:
        try:
            if isinstance(item, IterableBase):
                yield from flatten_to_ints(item)
            elif int(item) == item:
                yield item
        except TypeError as ex:
            raise ValueError(f"Expected {item} to be iterable or an integer.") from ex


def get_rng(seed: Rng) -> Generator:
    r"""
    Return a :class:`numpy.random.Generator`.

    Args:
        seed: An integer seed used to make a new generator, or a generator itself, or ``None`` for
            a static module-wide generator.

    Return:
        A :class:`numpy.random.Generator`.
    """
    if seed is None:
        return DEFAULT_RNG
    elif isinstance(seed, Generator):
        return seed
    return default_rng(seed)


def iter_along_axis(array: np.ndarray, axis: int) -> Iterable[np.ndarray]:
    r"""
    Yield slices of an array along a specified axis.

    Args:
        array: The array to return slices from.
        axis: The axis along which to iterate.
    Yields:
        Slices of ``array`` along ``axis``.
    """
    array = array.transpose((axis,) + tuple(idx for idx in range(array.ndim) if idx != axis))
    yield from array


def iter_as_shape(array: np.ndarray, shape: Shape) -> Iterable[np.ndarray]:
    r"""
    Yield slices of the last dimensions of ``array`` as though its shape begins with ``shape``.

    This functions exists as an aid for slice iteration when broadcasting rules are required:
    ``shape`` must agree with the first part of ``array.shape`` on all non-singleton dimensions.

    Args:
        array: An array whose first dimensions agree with ``shape`` on all non-singleton dimensions.
        shape: The implied desired shape of the first dimensions of ``array``.\
    
    Yields:
        Slices of the ``array.ndim - len(shape)`` last dimensions of ``array``.
    """
    pad = {idx for idx, (s1, s2) in enumerate(zip(shape, array.shape)) if s1 != s2}
    for idxs in product(*map(range, shape)):
        idxs = tuple(0 if idx in pad else i for idx, i in enumerate(idxs))
        yield array[idxs + (...,)]


def ndrange(shape: Shape, dtype: DTypeLike = int) -> np.ndarray:
    r"""
    Return an array ``a`` such that ``tuple(a[idx]) == idx`` for any tuple of non-negative integers
    dominated by the ``shape``.

    That is, all possible integer tuples of the given ``shape`` are generated into the last
    dimension of the returned array. This is equivalent to

    .. code-block::

        def ndrange(shape, dtype=int):
            return np.array(list, product(*map(range, shape)))), dtype=dtype).reshape(shape + (-1,))

    Args:
        shape: The dimensions.
        dtype: The dtype of the returned array.

    Returns:
        An array of shape ``shape + (len(shape),)``.
    """
    ret = np.empty(shape_tuple(shape, len(shape)), dtype=dtype)
    broadcast = tuple(None if i else slice(None, None, None) for i in range(len(shape)))
    for idx, dim in enumerate(shape):
        ret[..., idx] = np.arange(dim, dtype=dtype)[broadcast]
        broadcast = (None,) + broadcast[:-1]
    return ret


def shape_tuple(*shapes: Union[int, Iterable]) -> Shape:
    """
    Flatten the input into a single tuple of integers, preserving order.

    Args:
        shapes: Integers or iterables of integers, possibly nested.

    Returns:
        A tuple of integers.

    Raises:
        ValueError: If some (possibly nested) member of ``shapes`` is not an integer or iterable.
    """
    return tuple(flatten_to_ints(shapes))


_NOT_FOUND = object()


class cached_property_by_dim(property):
    r"""A descriptor that caches a method output per-class, per-num_qubits.

    This descriptor is similar to the :func:`functools.cached_property` descriptor, except that the
    cache lives on the owner rather than the instance. Moreover, if the owner has an attribute
    called ``num_qubits``, then the output is cached for each unique (stringified) value of this
    attribute. Hence, the property value is borg-like, and so that this descriptor can only be
    safely used in cases where some guarantee can be made that the output of the method depends on
    no more than the value of ``num_qubits``.ß

    ``property`` is used as the parent to get increase support for linting.
    """

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.addendum = False
        self.__doc__ = func.__doc__
        self.lock = RLock()

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
            #  do the hasattr on num_qubits here instead of in __get__ for performance
            self.addendum = hasattr(owner, "num_qubits")
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            # this descriptor was accessed directly by the owner
            return self

        try:
            cache = getattr(owner, "_cached_properties_by_dim", _NOT_FOUND)
            if cache is _NOT_FOUND:
                cache = {}
                setattr(owner, "_cached_properties_by_dim", cache)
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(owner).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None

        try:
            attrname = self.attrname + (str(instance.num_qubits) if self.addendum else "")
        except TypeError:
            # this is an edge case where the user manually instantiates the descriptor in a
            # non-standard way but forgets to call __set_name__
            raise TypeError("Must call __set_name__ first.") from None

        val = cache.get(attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(attrname, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance)
                    try:
                        cache[attrname] = val
                    except TypeError:
                        msg = (
                            f"The '__dict__' attribute on {type(owner).__name__!r} instance does "
                            f"not support item assignment for caching {attrname!r} property."
                        )
                        raise TypeError(msg) from None
        return val

    def __set__(self, obj, value):
        raise AttributeError(f"Attribute {self.attrname} cannot be set.")
