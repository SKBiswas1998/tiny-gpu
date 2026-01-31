# TinyGPU

A minimal GPU implementation in Verilog for learning how GPUs work, with Python interface and LLM inference.

## Features

- **SIMT Architecture**: Single Instruction, Multiple Threads (like CUDA)
- **Complete RTL**: Full GPU in 15 Verilog files
- **GDS Tape-out**: Actually fabricated to silicon!
- **Python API**: CUDA-like programming interface
- **LLM Inference**: Run GPT-2 at 10-12 tokens/sec

## Quick Start

### Run GPT-2 Inference
```bash
python python/gpt2_inference.py
```

### CUDA-like Programming
```python
from python.tinygpu import TinyGPU
import numpy as np

gpu = TinyGPU(num_cores=2, threads_per_block=4)

# Vector addition
a = np.array([1, 2, 3, 4], dtype=np.uint8)
b = np.array([5, 6, 7, 8], dtype=np.uint8)
c = gpu.vector_add(a, b)  # [6, 8, 10, 12]

# Matrix multiplication
A = np.array([[1, 2], [3, 4]], dtype=np.uint8)
B = np.array([[1, 2], [3, 4]], dtype=np.uint8)
C = gpu.matmul(A, B)  # [[7, 10], [15, 22]]
```

## Architecture
```
GPU
 ├── Dispatcher (distributes threads to cores)
 ├── Core 0
 │    ├── Scheduler (manages thread execution)
 │    ├── Fetcher (instruction fetch)
 │    ├── Decoder (instruction decode)
 │    ├── ALU (arithmetic operations)
 │    ├── LSU (load/store unit)
 │    └── Registers (register file)
 ├── Core 1
 │    └── ...
 └── Memory Controllers
```

## ISA (16-bit instructions)

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| 0011 | ADD | Add registers |
| 0100 | SUB | Subtract |
| 0101 | MUL | Multiply |
| 0110 | DIV | Divide |
| 0111 | LDR | Load from memory |
| 1000 | STR | Store to memory |
| 1001 | CONST | Load constant |
| 1111 | RET | End kernel |

## Performance

| Operation | Cycles |
|-----------|--------|
| Vector Add (4 elem) | 13 |
| Vector Mul (4 elem) | 13 |
| Matmul 2x2 | 41 |
| GPT-2 inference | 10-12 tok/s |

## Project Structure
```
tiny-gpu/
├── src/           # Verilog RTL
├── python/        # Python interface + LLM
├── test/          # Cocotb tests
├── gds/           # Silicon layout (GDS)
└── docs/          # Documentation
```

## License

MIT
