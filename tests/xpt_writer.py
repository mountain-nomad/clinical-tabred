"""Minimal SAS XPORT (v5) writer used ONLY by the tests to build synthetic NHANES files.

pandas can read XPT but cannot write it, so this helper writes numeric columns in
the same transport format the CDC uses. Missing values are written as SAS ".".
"""

import struct
from pathlib import Path

import numpy as np
import pandas as pd


def _pad(text: str, width: int) -> bytes:
    return text.encode("ascii").ljust(width)[:width]


def _header(name: str, numbers: str = "0" * 30) -> bytes:
    return _pad(f"HEADER RECORD*******{name:<8}HEADER RECORD!!!!!!!{numbers}", 80)


def ibm_float(value: float) -> bytes:
    """Convert an IEEE double to the 8-byte IBM mainframe float used by XPORT."""
    if value is None or np.isnan(value):
        return b"\x2e" + b"\x00" * 7  # SAS missing value "."
    if value == 0:
        return b"\x00" * 8
    sign = 0x80 if value < 0 else 0
    value, exponent = abs(float(value)), 0
    while value >= 1:
        value /= 16
        exponent += 1
    while value < 1 / 16:
        value *= 16
        exponent -= 1
    mantissa = round(value * 2**56)
    if mantissa >= 2**56:
        mantissa >>= 4
        exponent += 1
    return bytes([sign | (exponent + 64)]) + mantissa.to_bytes(7, "big")


def _namestr(index: int, name: str) -> bytes:
    record = struct.pack(">hhhh", 1, 0, 8, index + 1) + _pad(name, 8) + _pad(name, 40)
    record += _pad("", 8) + struct.pack(">hhh", 0, 0, 0) + b"\x00\x00" + _pad("", 8)
    record += struct.pack(">hhl", 0, 0, index * 8)
    return record.ljust(140, b"\x00")


def _pad_block(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((-len(data)) % 80)


def write_xpt(df: pd.DataFrame, path: Path, dataset_name: str = "DATA") -> None:
    """Write a DataFrame with numeric columns to an XPT file readable by pandas."""
    stamp = "01JAN20:00:00:00"
    out = _header("LIBRARY")
    out += _pad("SAS     SAS     SASLIB  9.4     X64_10PR" + " " * 24 + stamp, 80)
    out += _pad(stamp, 80)
    out += _header("MEMBER", "000000000000000001600000000140")
    out += _header("DSCRPTR")
    out += _pad(f"SAS     {dataset_name:<8}SASDATA 9.4     X64_10PR" + " " * 24 + stamp, 80)
    out += _pad(stamp + " " * 16 + dataset_name.ljust(40) + "DATA    ", 80)
    out += _header("NAMESTR", f"000000{len(df.columns):04d}" + "0" * 20)
    out += _pad_block(b"".join(_namestr(i, c) for i, c in enumerate(df.columns)), b"\x00")
    out += _header("OBS")
    rows = b"".join(ibm_float(v) for row in df.to_numpy(dtype=float) for v in row)
    out += _pad_block(rows, b" ")
    Path(path).write_bytes(out)
