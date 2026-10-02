from dataclasses import dataclass
from typing import Generic, TypeVar
import math

from .failures import ContractViolation, LowConfidence

T = TypeVar("T")


@dataclass(frozen=True)
class Uncertain(Generic[T]):
    value: T
    confidence: float
    calibrated: bool = False

    def __post_init__(self) -> None:
        confidence = self.confidence
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or math.isnan(confidence):
            raise ContractViolation(f"confidence must be a number, got {confidence!r}", expected="number", actual=confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ContractViolation("confidence must be between 0 and 1", expected=[0.0, 1.0], actual=confidence)

    def require_confidence(self, minimum: float, *, require_calibrated: bool = False) -> T:
        if require_calibrated and not self.calibrated:
            raise ContractViolation(
                "confidence is not calibrated",
                expected="calibrated", actual="uncalibrated", remediation="calibrate the confidence source",
            )
        if self.confidence < minimum:
            raise LowConfidence(
                f"confidence {self.confidence:.3f} is below required {minimum:.3f}",
                expected=minimum, actual=self.confidence,
            )
        return self.value
