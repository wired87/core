"""
Thalamic Event Classifier

Unified classification facade: BrainClassifier (rule + vector) first,
AIChatClassifier (LLM) fallback when confidence is low.
"""
from qbrain.core.thalamic_classifier.thalamic_event_classifier import (
    ThalamicEventClassifier,
)

__all__ = ["ThalamicEventClassifier"]
