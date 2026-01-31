"""
TinyGPU Unified Backend System
==============================
Auto-selects the fastest available backend (like TinyTPU).
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Union

class Backend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass
    
    @property
    @abstractmethod
    def device(self) -> str: pass
    
    @abstractmethod
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray: pass
    
    @abstractmethod
    def matmul_float(self, A: np.ndarray, B: np.ndarray) -> np.ndarray: pass
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def layer_norm(self, x, weight=None, bias=None, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        if weight is not None: x_norm = x_norm * weight
        if bias is not None: x_norm = x_norm + bias
        return x_norm
    
    def is_available(self) -> bool: return True


class NumpyBackend(Backend):
    @property
    def name(self): return "numpy"
    @property
    def device(self): return "cpu"
    
    def matmul(self, A, B):
        return np.matmul(A.astype(np.int32), B.astype(np.int32))
    
    def matmul_float(self, A, B):
        return np.matmul(A.astype(np.float32), B.astype(np.float32))


class PyTorchBackend(Backend):
    def __init__(self, device=None):
        self._torch = None
        self._device = None
        self._available = False
        try:
            import torch
            self._torch = torch
            if device is None:
                if torch.cuda.is_available():
                    self._device = torch.device('cuda')
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self._device = torch.device('mps')
                else:
                    self._device = torch.device('cpu')
            else:
                self._device = torch.device(device)
            if self._device.type == 'cpu':
                import os
                torch.set_num_threads(os.cpu_count() or 4)
            self._available = True
        except ImportError:
            pass
    
    @property
    def name(self): return "pytorch"
    @property
    def device(self): return str(self._device) if self._device else "unknown"
    def is_available(self): return self._available
    
    def matmul(self, A, B):
        torch = self._torch
        A_t = torch.from_numpy(A.astype(np.int32)).to(self._device)
        B_t = torch.from_numpy(B.astype(np.int32)).to(self._device)
        C_t = torch.matmul(A_t, B_t)
        return C_t.cpu().numpy() if self._device.type != 'cpu' else C_t.numpy()
    
    def matmul_float(self, A, B):
        torch = self._torch
        A_t = torch.from_numpy(A.astype(np.float32)).to(self._device)
        B_t = torch.from_numpy(B.astype(np.float32)).to(self._device)
        C_t = torch.matmul(A_t, B_t)
        return C_t.cpu().numpy() if self._device.type != 'cpu' else C_t.numpy()
    
    def softmax(self, x, axis=-1):
        torch = self._torch
        x_t = torch.from_numpy(x.astype(np.float32)).to(self._device)
        y_t = torch.nn.functional.softmax(x_t, dim=axis)
        return y_t.cpu().numpy() if self._device.type != 'cpu' else y_t.numpy()
    
    def relu(self, x):
        torch = self._torch
        x_t = torch.from_numpy(x.astype(np.float32)).to(self._device)
        y_t = torch.relu(x_t)
        return y_t.cpu().numpy() if self._device.type != 'cpu' else y_t.numpy()
    
    def gelu(self, x):
        torch = self._torch
        x_t = torch.from_numpy(x.astype(np.float32)).to(self._device)
        y_t = torch.nn.functional.gelu(x_t)
        return y_t.cpu().numpy() if self._device.type != 'cpu' else y_t.numpy()


def get_best_backend(verbose=True):
    if verbose: print("Detecting best backend...")
    
    # Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            b = PyTorchBackend('cuda')
            if b.is_available():
                if verbose: print(f"Selected: PyTorch CUDA ({torch.cuda.get_device_name(0)})")
                return b
    except: pass
    
    # Try PyTorch MPS
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            b = PyTorchBackend('mps')
            if b.is_available():
                if verbose: print("Selected: PyTorch MPS (Apple)")
                return b
    except: pass
    
    # Try PyTorch CPU
    try:
        b = PyTorchBackend('cpu')
        if b.is_available():
            if verbose: print("Selected: PyTorch CPU")
            return b
    except: pass
    
    # Fallback to NumPy
    if verbose: print("Selected: NumPy (fallback)")
    return NumpyBackend()


class TinyGPU:
    """TinyGPU with automatic backend selection."""
    
    def __init__(self, backend="auto"):
        if isinstance(backend, Backend):
            self._backend = backend
        elif backend == "auto":
            self._backend = get_best_backend(verbose=True)
        elif backend == "numpy":
            self._backend = NumpyBackend()
        elif backend == "pytorch":
            self._backend = PyTorchBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    @property
    def backend_name(self): return self._backend.name
    @property
    def device(self): return self._backend.device
    
    def matmul(self, A, B):
        return self._backend.matmul(np.asarray(A), np.asarray(B))
    
    def matmul_float(self, A, B):
        return self._backend.matmul_float(np.asarray(A), np.asarray(B))
    
    def softmax(self, x, axis=-1):
        return self._backend.softmax(np.asarray(x).astype(np.float32), axis)
    
    def relu(self, x):
        return self._backend.relu(np.asarray(x).astype(np.float32))
    
    def gelu(self, x):
        return self._backend.gelu(np.asarray(x).astype(np.float32))
    
    def layer_norm(self, x, weight=None, bias=None, eps=1e-5):
        return self._backend.layer_norm(np.asarray(x).astype(np.float32), weight, bias, eps)
    
    def benchmark(self, size=512, iterations=10):
        A = np.random.randint(-128, 127, (size, size), dtype=np.int8)
        B = np.random.randint(-128, 127, (size, size), dtype=np.int8)
        self.matmul(A, B)  # warmup
        start = time.perf_counter()
        for _ in range(iterations): self.matmul(A, B)
        elapsed = time.perf_counter() - start
        return {
            'backend': self.backend_name, 'device': self.device,
            'size': size, 'time_ms': elapsed/iterations*1000,
            'gops': (2*size**3*iterations)/elapsed/1e9
        }
    
    def __repr__(self):
        return f"TinyGPU(backend={self.backend_name!r}, device={self.device!r})"


if __name__ == "__main__":
    print("=" * 60)
    print("TINYGPU UNIFIED BACKEND")
    print("=" * 60)
    
    gpu = TinyGPU(backend="auto")
    print(f"\n{gpu}\n")
    
    print("Benchmarking...")
    for size in [128, 256, 512, 1024]:
        r = gpu.benchmark(size=size, iterations=5)
        print(f"  {size}x{size}: {r['time_ms']:.2f}ms, {r['gops']:.2f} GOPS")
    
    print("\nTesting operations...")
    A = np.random.randint(-128, 127, (64, 128), dtype=np.int8)
    B = np.random.randint(-128, 127, (128, 64), dtype=np.int8)
    C = gpu.matmul(A, B)
    C_ref = np.matmul(A.astype(np.int32), B.astype(np.int32))
    print(f"  matmul: {'PASS' if np.array_equal(C, C_ref) else 'FAIL'}")
    
    x = np.random.randn(4, 10).astype(np.float32)
    y = gpu.softmax(x)
    print(f"  softmax: {'PASS' if np.allclose(y.sum(axis=-1), 1.0) else 'FAIL'}")
    
    print("\n" + "=" * 60)
    print("DONE!")
