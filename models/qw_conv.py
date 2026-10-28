"""
Backward compatibility forwarder for QuasarConv.
"""
from models.quasar_conv import QuasarConv as QWConv
from models.quasar_conv import QuasarConv

__all__ = ["QWConv", "QuasarConv"]
