"""Conservative, language-independent normalization using no external entity data."""

from __future__ import annotations

import math
import re
from typing import Any

from unidecode import unidecode

_PUNCTUATION = re.compile(r"[^a-z0-9]+")
_NULL_TOKEN = re.compile(r"<\s*(?:null|nan|na|none)\s*>", re.IGNORECASE)


def normalize_text(text: Any) -> str:
    """Transliterate to Latin, casefold, and normalize punctuation; keep all digits.

    Explicit null placeholders disappear. A bounded cache may be useful in callers,
    but this function deliberately keeps no global cache of the million-row corpus.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            if math.isnan(text):
                return ""
        except (TypeError, ValueError):
            pass
        text = str(text)
    text = _NULL_TOKEN.sub(" ", text)
    if text.strip().casefold() in {"", "nan", "none", "null", "<na>"}:
        return ""
    text = unidecode(text, errors="ignore").casefold().replace("&", " and ")
    return _PUNCTUATION.sub(" ", text).strip()
