"""GoodHart Monitor — independent verification records for deployed clinical models.

The contract is a scored stream, not a model: what the deployed system emitted
and what later turned out to be true. That is what a hospital actually has, and
checking outputs rather than internals is what makes the check independent.
"""
from .contract import ScoredStream, ContractError, validate, load as load_stream
from .card import ModelCard, Claim, CardError, load as load_card
from .config import Config, load as load_config, DEFAULT as DEFAULT_CONFIG
from .record import build as build_record, canonical, sha256, SCHEMA
from .render import to_markdown
from .stats import HOLDS, FAILS, INDETERMINATE, NOT_APPLICABLE

__version__ = "0.1.0"
__all__ = [
    "ScoredStream", "ContractError", "validate", "load_stream",
    "ModelCard", "Claim", "CardError", "load_card",
    "Config", "load_config", "DEFAULT_CONFIG",
    "build_record", "canonical", "sha256", "SCHEMA", "to_markdown",
    "HOLDS", "FAILS", "INDETERMINATE", "NOT_APPLICABLE", "__version__",
]
