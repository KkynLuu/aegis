"""
Backward compatibility forwarder for QuasarClassifier.
"""
from models.quasar_classifier import QuasarClassifier as QWClassifier
from models.quasar_classifier import QuasarClassifier

__all__ = ["QWClassifier", "QuasarClassifier"]
