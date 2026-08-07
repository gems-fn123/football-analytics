"""Entity resolution: stable club IDs across seasons, sources, and renames.

This is a first-class module, not a scraper afterthought, because Indonesian
football maximises name entropy: mononyms, nicknames, club renames (Bali United
was Persisam Putra), sponsor suffixes (Bhayangkara Presisi), transliteration
variance, and short forms ("Persib" vs "Persib Bandung"). A wrong club ID
corrupts every aggregate downstream; a null one is honest and visible.

Resolution order, strictest first:
1. DO_NOT_MERGE - hard negative pairs; similarity never overrides these.
2. Manual overrides + curated alias registry (exact, after normalisation).
3. Conservative fuzzy: unique prefix/containment match with all rivals far
   worse - and only when the pair is not forbidden.
4. Abstain: return None. Unresolved strings surface in `unresolved()` for
   curation, they are never silently guessed.

The registry lives in code (reviewable, versioned, testable). Additions come
from `unresolved()` triage, not from loosening the matcher.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical registry. canonical_id -> display name + aliases (all lowercase,
# normalised). Aliases must be UNIQUE across clubs - enforced at import time.
# ---------------------------------------------------------------------------

CLUBS: dict[str, dict] = {
    "persib": {"name": "Persib Bandung", "aliases": ["persib", "persib bandung"]},
    "persija": {"name": "Persija Jakarta", "aliases": ["persija", "persija jakarta"]},
    "persebaya": {"name": "Persebaya Surabaya", "aliases": ["persebaya", "persebaya surabaya", "persatuan sepak bola surabaya"]},
    "arema": {"name": "Arema", "aliases": ["arema", "arema fc", "arema malang"]},
    "psm": {"name": "PSM Makassar", "aliases": ["psm", "psm makassar", "persatuan sepakbola makassar"]},
    "bali-united": {
        "name": "Bali United",
        "aliases": ["bali united", "bali united fc", "persisam putra", "putra samarinda"],
    },
    "borneo": {
        "name": "Borneo Samarinda",
        "aliases": ["borneo", "borneo fc", "borneo samarinda", "pusamania borneo"],
    },
    "bhayangkara": {
        "name": "Bhayangkara",
        "aliases": [
            "bhayangkara",
            "bhayangkara fc",
            "bhayangkara presisi",
            "bhayangkara solo",
            "bhayangkara surabaya united",
        ],
    },
    "dewa-united": {
        "name": "Dewa United",
        "aliases": ["dewa united", "dewa united banten", "dewa united fc", "martapura dewa united"],
    },
    "persik-kediri": {"name": "Persik Kediri", "aliases": ["persik", "persik kediri"]},
    "persik-kendal": {"name": "Persik Kendal", "aliases": ["persik kendal"]},
    "psps-riau": {"name": "PSPS Riau", "aliases": ["psps", "psps riau", "psps pekanbaru"]},
    "persita": {"name": "Persita Tangerang", "aliases": ["persita", "persita tangerang"]},
    "persipura": {"name": "Persipura Jayapura", "aliases": ["persipura", "persipura jayapura"]},
    "psis": {"name": "PSIS Semarang", "aliases": ["psis", "psis semarang"]},
    "pss": {"name": "PSS Sleman", "aliases": ["pss", "pss sleman", "persatuan sepakbola sleman"]},
    "persis": {"name": "Persis Solo", "aliases": ["persis", "persis solo"]},
    "barito-putera": {"name": "Barito Putera", "aliases": ["barito putera", "ps barito putera"]},
    "madura-united": {"name": "Madura United", "aliases": ["madura united", "madura united fc"]},
    "persela": {"name": "Persela Lamongan", "aliases": ["persela", "persela lamongan", "persatuan sepak bola lamongan"]},
    "persiraja": {"name": "Persiraja Banda Aceh", "aliases": ["persiraja", "persiraja banda aceh"]},
    "semen-padang": {"name": "Semen Padang", "aliases": ["semen padang", "semen padang fc"]},
    "sriwijaya": {"name": "Sriwijaya", "aliases": ["sriwijaya", "sriwijaya fc"]},
    "psms": {"name": "PSMS Medan", "aliases": ["psms", "psms medan"]},
    "kalteng-putra": {"name": "Kalteng Putra", "aliases": ["kalteng putra", "kalteng putra fc"]},
    "tira-persikabo": {
        "name": "Persikabo 1973",
        "aliases": ["persikabo 1973", "tira persikabo", "ps tira", "tira-persikabo", "tni"],
    },
    "mitra-kukar": {"name": "Mitra Kukar", "aliases": ["mitra kukar"]},
    "perseru": {
        # Perseru Serui sold its licence and became Badak Lampung in 2019 -
        # one entity, two homes, three names.
        "name": "Badak Lampung",
        "aliases": ["perseru", "perseru serui", "badak lampung", "perseru badak lampung"],
    },
    "rans": {
        "name": "RANS Nusantara",
        "aliases": ["rans nusantara", "rans cilegon", "rans nusantara fc"],
    },
    "psim": {"name": "PSIM Yogyakarta", "aliases": ["psim", "psim yogyakarta"]},
    "persiba-balikpapan": {
        "name": "Persiba Balikpapan",
        "aliases": ["persiba", "persiba balikpapan"],
    },
    "persegres": {
        "name": "Gresik United",
        "aliases": ["persegres", "gresik united", "persegres gresik united"],
    },
    "garudayaksa": {"name": "Garudayaksa", "aliases": ["garudayaksa"]},
    "java-united": {"name": "Java United", "aliases": ["java united"]},
    "psbs": {"name": "PSBS Biak", "aliases": ["psbs", "psbs biak"]},
    "malut-united": {"name": "Malut United", "aliases": ["malut united", "malut united fc", "maluku utara united"]},
    "persijap": {"name": "Persijap Jepara", "aliases": ["persijap", "persijap jepara"]},
}

# Pairs that must NEVER merge, whatever any matcher thinks. Both orders hold.
DO_NOT_MERGE: set[tuple[str, str]] = {
    ("persik", "persik kendal"),
    ("persik kediri", "persik kendal"),
    ("persipura", "persipura kendari"),  # Liga 3 namesake
    ("persis", "psis"),  # one edit apart, different clubs
    ("persib", "persiba"),  # Persiba Balikpapan
    ("persib", "persibat"),
    ("persija", "persijap"),  # Jakarta vs Jepara, one letter apart
    ("persita", "persiba"),
}


def normalise(name: str) -> str:
    """Lowercase, strip accents/punctuation/'FC' noise - the comparison form."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[.\-/']", " ", s)
    s = re.sub(r"\b(fc|cf|ps)\b\.?", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _forbidden(a: str, b: str) -> bool:
    return (a, b) in DO_NOT_MERGE or (b, a) in DO_NOT_MERGE


@dataclass
class ClubResolver:
    """Resolve raw club strings to canonical IDs; abstain when unsure."""

    overrides: dict[str, str] = field(default_factory=dict)  # normalised raw -> canonical_id
    _alias_index: dict[str, str] = field(init=False)
    _misses: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        index: dict[str, str] = {}
        for cid, spec in CLUBS.items():
            for alias in spec["aliases"]:
                key = normalise(alias)
                if key in index and index[key] != cid:
                    raise ValueError(f"alias {key!r} claimed by {index[key]} and {cid}")
                index[key] = cid
        self._alias_index = index
        for raw, cid in self.overrides.items():
            if cid not in CLUBS:
                raise ValueError(f"override target {cid!r} is not a registered club")
            self._alias_index[normalise(raw)] = cid

    def resolve(self, raw: str) -> str | None:
        """Canonical club id, or None. None means 'curate me', never 'guess'."""
        key = normalise(raw)
        if not key:
            return None
        hit = self._alias_index.get(key)
        if hit:
            return hit
        # Youth/reserve/women's sides are distinct entities; a senior-club
        # containment match would silently pool their stats into the seniors'.
        if re.search(r"\b(u ?\d{2}|ii|junior|women|putri|academy)\b", key):
            self._misses.add(raw)
            return None

        # Conservative fuzzy tier: containment against aliases, unique winner
        # only, and never across a DO_NOT_MERGE pair.
        candidates = set()
        for alias, cid in self._alias_index.items():
            if _forbidden(key, alias):
                continue
            if alias.startswith(key + " ") or key.startswith(alias + " "):
                candidates.add(cid)
        if len(candidates) == 1:
            return candidates.pop()

        self._misses.add(raw)
        return None

    def display_name(self, canonical_id: str) -> str:
        return CLUBS[canonical_id]["name"]

    def unresolved(self) -> list[str]:
        """Raw strings that abstained this session - the curation worklist."""
        return sorted(self._misses)


def resolve_clubs(
    df: pd.DataFrame, column: str, resolver: ClubResolver | None = None
) -> pd.DataFrame:
    """Add club_id (nullable string) next to a raw name column."""
    resolver = resolver or ClubResolver()
    out = df.copy()
    base = re.sub(r"_(raw|name)$", "", column)
    out[f"{base}_id"] = df[column].map(lambda v: resolver.resolve(v)).astype("string")
    return out
