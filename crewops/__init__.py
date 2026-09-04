"""Crew Ops Advisor -- a conversational decision aid for airline Crew Control.

The model translates. The kernel decides. Nothing else touches a number.
"""
__version__ = "1.0.0"

from .data import load, load_cached, Snapshot          # noqa: F401
from .kernel import check_cover, cover_options, solve_joint  # noqa: F401
from .agent import Advisor                              # noqa: F401
