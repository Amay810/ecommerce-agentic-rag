"""Minimal GRPO integration surface for the frozen tau3 retail pilot.

The package intentionally contains adapters and provenance/validation code,
not a replacement optimizer or rollout engine.  VERL is an external runtime
dependency on the NSCC training node.
"""

from .config import FrozenTau3GRPOConfig

__all__ = ["FrozenTau3GRPOConfig"]
