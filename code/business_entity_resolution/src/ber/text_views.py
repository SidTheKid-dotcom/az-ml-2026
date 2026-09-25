"""Canonical, field-preserving text views for neural entity matching."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


_JUNK = re.compile(r"<null>|##|>>|<<", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False
    return False


def _latin_letter(character: str) -> bool:
    return character.isalpha() and "LATIN" in unicodedata.name(character, "")


def canonicalize(value: Any) -> str:
    """Normalize model text without transliterating or damaging non-Latin scripts.

    Latin letters are decomposed and their combining marks removed. Combining
    marks belonging to every other script are retained.
    """
    if _is_null(value):
        return ""
    text = _JUNK.sub(" ", str(value)).casefold()
    output: list[str] = []
    previous_latin = False
    for character in text:
        if _latin_letter(character):
            output.extend(part for part in unicodedata.normalize("NFD", character)
                          if not unicodedata.combining(part))
            previous_latin = True
        elif unicodedata.combining(character) and previous_latin:
            continue
        else:
            output.append(character)
            if not unicodedata.combining(character):
                previous_latin = False
    return _SPACE.sub(" ", unicodedata.normalize("NFC", "".join(output))).strip()


def record_text(name: Any, address: Any, *, canonical: bool = True,
                field_tagged: bool = True) -> str:
    """Build one record view while retaining an explicit name/address boundary."""
    if canonical:
        name, address = canonicalize(name), canonicalize(address)
    else:
        name = "" if _is_null(name) else str(name)
        address = "" if _is_null(address) else str(address)
    if field_tagged:
        return f"[NAME] {name} [ADDRESS] {address}"
    return f"{name} | {address}"


def record_texts(frame: pd.DataFrame, *, canonical: bool = True,
                 field_tagged: bool = True) -> list[str]:
    """Build record views in frame order; IDs and country are never included."""
    required = {"business_name", "business_address"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Record text needs columns: {sorted(missing)}")
    return [record_text(name, address, canonical=canonical, field_tagged=field_tagged)
            for name, address in zip(frame.business_name, frame.business_address)]
