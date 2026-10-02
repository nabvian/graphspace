from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Basis(str, Enum):
    DECLARED = "declared"
    PROVEN = "proven"
    INFERRED = "inferred"
    ESTIMATED = "estimated"
    RUNTIME_CHECKED = "runtime_checked"
    MEASURED = "measured"
    BACKEND_REPORTED = "backend_reported"


@dataclass(frozen=True)
class Claim:
    name: str
    value: Any
    basis: Basis


def find_claim(claims: Iterable[Claim], name: str) -> Claim:
    for claim in claims:
        if claim.name == name:
            return claim
    raise KeyError(name)
