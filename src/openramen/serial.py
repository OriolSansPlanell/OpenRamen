"""JSON encoding of literal values stored in graph files.

JSON only knows null, booleans, numbers, strings, lists and string-keyed
objects.  Everything else a pin literal can hold is written as a one-key
object whose key starts with ``$``, for example ``{"$complex": [1.0, 2.0]}``.
"""

from __future__ import annotations

import base64
import math
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

__all__ = ["SerializationError", "encode_value", "decode_value"]


class SerializationError(ValueError):
    """A value cannot be written to, or read from, a graph file."""


_FLOAT_NAMES = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def encode_value(value: Any) -> Any:
    """Turn a literal into plain JSON data.  Inverse of :func:`decode_value`."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        if isinstance(value, np.str_):
            return str(value)
        if isinstance(value, (np.bool_, np.integer, np.floating, np.complexfloating)):
            return encode_value(value.item())
        raise SerializationError(
            f"a NumPy scalar of type {type(value).__name__} cannot be saved in a graph")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"$float": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, complex):
        return {"$complex": [encode_value(value.real), encode_value(value.imag)]}
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, PurePath):
        flavour = "windows" if isinstance(value, PureWindowsPath) else "posix"
        return {"$path": str(value), "flavour": flavour}
    if isinstance(value, list):
        return [encode_value(v) for v in value]
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return {"$tuple": [encode_value(v) for v in value]}
    if isinstance(value, dict):
        if all(isinstance(k, str) and not k.startswith("$") for k in value):
            return {k: encode_value(v) for k, v in value.items()}
        return {"$dict": [[encode_value(k), encode_value(v)] for k, v in value.items()]}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise SerializationError(
                f"an array of dtype {value.dtype} cannot be saved in a graph")
        return {"$ndarray": {
            "dtype": value.dtype.name,
            "shape": list(value.shape),
            "data": encode_value(value.ravel().tolist()),
        }}
    raise SerializationError(
        f"a value of type {type(value).__name__} cannot be saved in a graph; "
        "produce it with a node instead of storing it as a literal")


def decode_value(data: Any) -> Any:
    """Rebuild a literal from the JSON data written by :func:`encode_value`."""
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [decode_value(v) for v in data]
    if not isinstance(data, dict):
        raise SerializationError(f"unexpected JSON value of type {type(data).__name__}")
    tagged = [k for k in data if isinstance(k, str) and k.startswith("$")]
    if not tagged:
        return {k: decode_value(v) for k, v in data.items()}
    try:
        return _decode_tagged(data, tagged)
    except SerializationError:
        raise
    except Exception as exc:
        raise SerializationError(f"malformed literal {data!r}: {exc}") from exc


def _decode_tagged(data: dict, tagged: list[str]) -> Any:
    if len(tagged) != 1:
        raise SerializationError(f"malformed literal {data!r}")
    tag = tagged[0]
    body = data[tag]
    if tag == "$float":
        return _FLOAT_NAMES[body]
    if tag == "$complex":
        re, im = body
        return complex(float(decode_value(re)), float(decode_value(im)))
    if tag == "$bytes":
        return base64.b64decode(body.encode("ascii"), validate=True)
    if tag == "$path":
        if not isinstance(body, str):
            raise SerializationError(f"malformed literal {data!r}")
        cls = PureWindowsPath if data.get("flavour") == "windows" else PurePosixPath
        # A usable ``Path`` when the flavour matches this machine, else a pure one.
        native = Path(body)
        return native if isinstance(native, cls) else cls(body)
    if tag == "$tuple":
        return tuple(decode_value(v) for v in body)
    if tag == "$dict":
        return {_hashable(decode_value(k)): decode_value(v) for k, v in body}
    if tag == "$ndarray":
        dtype = np.dtype(body["dtype"])
        if dtype.kind not in "biufc":
            raise SerializationError(f"unsupported array dtype {dtype}")
        flat = decode_value(body["data"])
        return np.array(flat, dtype=dtype).reshape([int(n) for n in body["shape"]])
    raise SerializationError(f"unknown literal tag {tag!r}")


def _hashable(key: Any) -> Any:
    try:
        hash(key)
    except TypeError:
        raise SerializationError(f"dictionary key {key!r} is not hashable") from None
    return key
