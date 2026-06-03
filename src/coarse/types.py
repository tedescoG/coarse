"""Shared type aliases used across coarse modules."""

from typing import Hashable, TypeAlias

Block: TypeAlias = frozenset[int]
EnvKey: TypeAlias = Hashable
