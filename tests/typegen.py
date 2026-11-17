"""Seeded random generators for property-style tests (no extra dependency)."""

from __future__ import annotations

import dataclasses
import enum
import random
from pathlib import PurePosixPath
from typing import Any, NamedTuple

import numpy as np

from openramen.types import (
    ANY, BOOL, BYTES, COMPLEX, FLOAT, INT, NONE, PATH, STR,
    ArrayT, CallableT, DictT, EnumT, IteratorT, ListT, ObjectT, RecordT,
    TupleT, Type, VarT, union,
)

LEAVES: list[Type] = [
    BOOL, INT, FLOAT, COMPLEX, STR, BYTES, PATH, NONE, ANY,
    ArrayT(), ArrayT("float64", 2), ArrayT("uint8", None), ArrayT(None, 3),
    ArrayT("int32", 1), ArrayT("complex128", 0),
    EnumT("pkg.Color", ("red", "green")), EnumT("pkg.Empty", ()),
    ObjectT("builtins:object"), ObjectT("collections:OrderedDict"),
    RecordT("pkg.Point", (("x", FLOAT), ("y", FLOAT))), RecordT("pkg.Unit", ()),
]

VARS: list[Type] = [VarT("T"), VarT("U"), ArrayT(VarT("D"), VarT("N")),
                    ArrayT(VarT("D"), 2), ArrayT("float32", VarT("N"))]


def random_type(rng: random.Random, depth: int = 0, variables: bool = True,
                vars_everywhere: bool = True) -> Type:
    """A random type up to three levels deep.

    ``vars_everywhere=False`` keeps variables out of unions and records, the
    two places :func:`openramen.types.unify` does not solve.
    """
    def sub(inside_opaque: bool = False) -> Type:
        allow = variables and (vars_everywhere or not inside_opaque)
        return random_type(rng, depth + 1, allow, vars_everywhere)

    if depth >= 3 or rng.random() < 0.35:
        pool = LEAVES + (VARS if variables else [])
        return rng.choice(pool)
    kind = rng.choice(["list", "tuple", "vtuple", "empty", "dict", "union",
                       "callable", "ucallable", "iterator", "record"])
    if kind == "list":
        return ListT(sub())
    if kind == "tuple":
        return TupleT(tuple(sub() for _ in range(rng.randint(1, 3))))
    if kind == "vtuple":
        return TupleT((sub(),), variadic=True)
    if kind == "empty":
        return TupleT(())
    if kind == "dict":
        return DictT(rng.choice([STR, INT, ANY]), sub())
    if kind == "union":
        return union(*(sub(inside_opaque=True) for _ in range(rng.randint(2, 3))))
    if kind == "callable":
        return CallableT(tuple(sub() for _ in range(rng.randint(0, 2))), sub())
    if kind == "ucallable":
        return CallableT(None, sub())
    if kind == "iterator":
        return IteratorT(sub())
    fields = tuple((f"f{i}", sub(inside_opaque=True)) for i in range(rng.randint(1, 3)))
    return RecordT("pkg.Rec", fields)


class Color(enum.Enum):
    RED = 1
    GREEN = 2


@dataclasses.dataclass
class Point:
    x: float
    y: float


class Pair(NamedTuple):
    first: int
    second: str


class Opaque:
    pass


def random_value(rng: random.Random, depth: int = 0) -> Any:
    """A random Python value of the kinds that travel on wires."""
    scalars = [
        lambda: None, lambda: rng.random() < 0.5, lambda: rng.randint(-5, 5),
        lambda: rng.uniform(-1, 1), lambda: complex(rng.random(), rng.random()),
        lambda: rng.choice(["", "a", "ramen"]), lambda: b"\x00\x01",
        lambda: PurePosixPath("/tmp/x"), lambda: np.float32(1.5),
        lambda: np.int16(3), lambda: np.bool_(True), lambda: np.complex64(1j),
        lambda: np.zeros((2, 3)), lambda: np.arange(4, dtype="uint8"),
        lambda: np.array(1.0), lambda: np.array(["a", "bc"]),
        lambda: np.array(["2020-01-01"], dtype="datetime64[ns]"),
        lambda: rng.choice(list(Color)), lambda: Point(1.0, 2.0),
        lambda: Pair(1, "a"), lambda: Opaque(), lambda: len,
        lambda: iter([1, 2]),
    ]
    if depth >= 2 or rng.random() < 0.5:
        return rng.choice(scalars)()
    kind = rng.choice(["list", "mixed", "tuple", "long", "dict", "empty"])
    if kind == "list":
        proto = rng.choice(scalars)
        return [proto() for _ in range(rng.randint(1, 4))]
    if kind == "mixed":
        return [random_value(rng, depth + 1) for _ in range(rng.randint(1, 6))]
    if kind == "tuple":
        return tuple(random_value(rng, depth + 1) for _ in range(rng.randint(0, 3)))
    if kind == "long":
        return tuple(rng.randint(0, 9) for _ in range(20))
    if kind == "dict":
        return {rng.choice(["a", "b", "c"]): random_value(rng, depth + 1)
                for _ in range(rng.randint(1, 3))}
    return rng.choice([[], {}, ()])
