"""
IBM Quantum Configuration & Authentication.
Stores your API credentials locally for when you are ready to execute.
"""

import os

# Your IBM Quantum Platform Token (kept locally in your project):
IBM_QUANTUM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", "xSCknw4be-bXcLfpyYSvGJsn6kCj1Q9WoI41i5hXYTQ-")
IBM_QUANTUM_CHANNEL = "ibm_quantum_platform"

def get_service():
    """Initializes and returns the authenticated QiskitRuntimeService."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel=IBM_QUANTUM_CHANNEL, token=IBM_QUANTUM_TOKEN)
