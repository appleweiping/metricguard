"""Explicit and reproducible text normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class TextNormalizer:
    """A small, serializable normalization policy.

    Defaults are intentionally conservative: normalize Unicode and surrounding
    whitespace, but preserve case and punctuation.
    """

    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"
    lowercase: bool = False
    collapse_whitespace: bool = True
    strip_punctuation: bool = False

    def __post_init__(self) -> None:
        if self.unicode_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError(f"unsupported Unicode normalization form: {self.unicode_form}")
        for name in ("lowercase", "collapse_whitespace", "strip_punctuation"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")

    def __call__(self, value: object) -> str:
        """Normalize a string without implicit JSON-type coercion."""

        if not isinstance(value, str):
            raise TypeError("text metrics require string reference and prediction values")
        text = unicodedata.normalize(self.unicode_form, value).strip()
        if self.lowercase:
            text = text.casefold()
        if self.strip_punctuation:
            text = _PUNCTUATION.sub(" ", text)
        if self.collapse_whitespace:
            text = _WHITESPACE.sub(" ", text).strip()
        return text

    def tokenize(self, value: object) -> tuple[str, ...]:
        """Return whitespace-delimited normalized tokens."""

        normalized = self(value)
        return tuple(normalized.split()) if normalized else ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible policy description."""

        return {
            "unicode_form": self.unicode_form,
            "lowercase": self.lowercase,
            "collapse_whitespace": self.collapse_whitespace,
            "strip_punctuation": self.strip_punctuation,
        }
