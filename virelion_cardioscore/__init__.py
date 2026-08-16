"""
Virelion CardioScore
====================

CiPA-aligned cardiotoxicity risk scoring for human iPSC-CM MEA
field-potential recordings.

License: GPL-3.0-or-later
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("virelion-cardioscore")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

from virelion_cardioscore.analysis.cipa_scoring import CardioScoreEngine
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.io.synthetic import load_synthetic_dataset, generate_synthetic_mea

__all__ = [
    "__version__",
    "CardioScoreEngine",
    "CardioScorePipeline",
    "load_synthetic_dataset",
    "generate_synthetic_mea",
]
