"""Prediction modules for tennis betting."""

from .ensemble import EnsemblePredictor
from .upcoming import UpcomingMatchPredictor

__all__ = ['EnsemblePredictor', 'UpcomingMatchPredictor']
