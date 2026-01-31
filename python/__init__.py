"""
TinyGPU - Graphics Processing Unit
==================================
SIMT architecture with CUDA-like programming model.
Auto-selects fastest backend (PyTorch > NumPy).
"""

__version__ = "0.2.0"

from .unified_backend import TinyGPU, get_best_backend
from .unified_backend import Backend, NumpyBackend, PyTorchBackend
from .tinygpu import GPUSimulator, Assembler, TinyGPU as TinyGPUSim

__all__ = [
    "TinyGPU",
    "TinyGPUSim",
    "GPUSimulator",
    "Assembler",
    "get_best_backend",
    "Backend",
    "NumpyBackend",
    "PyTorchBackend",
]
