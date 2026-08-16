"""The four sections of a verification record.

Each is a pure function of (stream, card, config) -> dict, so each is testable
without a model, a corpus or a filesystem. They are separate modules because a
committee reads them separately and because a bug in one must not be able to
quietly change another.
"""
from .acceptance import acceptance
from .work import work, entity_table
from .timing import timing
from .drift import drift
from .subgroups import subgroups

__all__ = ["acceptance", "work", "timing", "drift", "subgroups", "entity_table"]
