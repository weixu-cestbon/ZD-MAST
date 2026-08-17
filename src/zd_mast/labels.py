"""Explicit AST label normalization for the frozen ZD-MAST task panel."""

from __future__ import annotations

import unicodedata


ANTIBIOTIC_ALIASES: dict[str, frozenset[str]] = {
    "oxacillin": frozenset({"苯唑西林", "oxacillin"}),
    "levofloxacin": frozenset({"左氧氟沙星", "levofloxacin"}),
    "gentamicin": frozenset({"庆大霉素", "gentamicin"}),
    "cefepime": frozenset({"头孢吡肟", "头孢吡肟(马斯平)", "cefepime"}),
    "ceftriaxone": frozenset({"头孢曲松", "ceftriaxone"}),
    "ceftazidime": frozenset({"头孢他啶", "ceftazidime"}),
    "ciprofloxacin": frozenset({"环丙沙星", "ciprofloxacin"}),
}


def normalize_ast_text(value: object) -> str:
    """Normalize spacing and Unicode width without changing drug semantics."""

    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).strip().casefold()


def map_core_antibiotic(value: object) -> str:
    """Map only explicit aliases from the historical ten-task panel.

    Exact matching is intentional. In particular, ceftazidime-avibactam must
    never be collapsed into ceftazidime simply because its name contains the
    ceftazidime substring.
    """

    text = normalize_ast_text(value)
    for canonical_name, aliases in ANTIBIOTIC_ALIASES.items():
        if text in aliases:
            return canonical_name
    return ""
