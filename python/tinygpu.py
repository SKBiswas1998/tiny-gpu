"""
TinyGPU Python Interface - FIXED
================================
"""

import numpy as np
from typing import List
from dataclasses import dataclass
from enum import IntEnum

class Opcode(IntEnum):
    NOP   = 0b0000
    BRnzp = 0b0001
    CMP   = 0b0010
    ADD   = 0b0011
    SUB   = 0b0100
    MUL   = 0b0101
    DIV   = 0b0110
    LDR   = 0b0111
    STR   = 0b1000
    CONST = 0b1001
    RET   = 0b1111

class SpecialReg(IntEnum):
    BLOCK_IDX  = 0b1110
    BLOCK_DIM  = 0b1101
    THREAD_IDX = 0b1111

class Assembler:
    def _reg(self, r: str) -> int:
        r = r.strip().upper()
        if r == '%BLOCKIDX': return SpecialReg.BLOCK_IDX
        elif r == '%BLOCKDIM': return SpecialReg.BLOCK_DIM
        elif r == '%THREADIDX': return SpecialReg.THREAD_IDX
        elif r.startswith('R'): return int(r[1:])
        else: raise ValueError(f"Invalid register: {r}")
    
    def encode(self, op, rd=0, rs1=0, rs2=0, imm=0) -> int:
        if op == Opcode.CONST:
            return (op << 12) | (rd << 8) | (imm & 0xFF)
        elif op == Opcode.BRnzp:
            return (op << 12) | (imm & 0xFFF)
        elif op == Opcode.STR:
            # STR: rd=data, rs1=addr → store data at addr
            return (op << 12) | (rd << 8) | (rs1 << 4)
        else:
            return (op << 12) | (rd << 8) | (rs1 << 4) | rs2
    
    def NOP(self): return self.encode(Opcode.NOP)
    def ADD(self, rd, rs1, rs2): return self.encode(Opcode.ADD, self._reg(rd), self._reg(rs1), self._reg(rs2))
    def SUB(self, rd, rs1, rs2): return self.encode(Opcode.SUB, self._reg(rd), self._reg(rs1), self._reg(rs2))
    def MUL(self, rd, rs1, rs2): return self.encode(Opcode.MUL, self._reg(rd), self._reg(rs1), self._reg(rs2))
    def DIV(self, rd, rs1, rs2): return self.encode(Opcode.DIV, self._reg(rd), self._reg(rs1), self._reg(rs2))
    def LDR(self, rd, rs): return self.encode(Opcode.LDR, self._reg(rd), self._reg(rs), 0)
    def STR(self, rs_addr, rs_data): 
        # STR addr_reg, data_reg → mem[addr_reg] = data_reg
        return self.encode(Opcode.STR, rd=self._reg(rs_data), rs1=self._reg(rs_addr))
    def CONST(self, rd, imm): return self.encode(Opcode.CONST, self._reg(rd), imm=imm)
    def CMP(self, rs1, rs2): return self.encode(Opcode.CMP, 0, self._reg(rs1), self._reg(rs2))
    def BRn(self, offset): return self.encode(Opcode.BRnzp, imm=(0b100 << 8) | (offset & 0xFF))
    def RET(self): return self.encode(Opcode.RET)

@dataclass
class ThreadState:
    thread_id: int
    block_id: int
    block_dim: int
    registers: List[int]
    pc: int = 0
    done: bool = False
    condition: int = 0

class GPUSimulator:
    def __init__(self, num_cores: int = 2, threads_per_block: int = 4):
        self.num_cores = num_cores
        self.threads_per_block = threads_per_block
        self.program_mem = [0] * 256
        self.data_mem = [0] * 256
        self.threads: List[ThreadState] = []
        self.cycles = 0
    
    def load_program(self, program: List[int]):
        for i, instr in enumerate(program):
            self.program_mem[i] = instr
    
    def load_data(self, data: List[int], offset: int = 0):
        for i, val in enumerate(data):
            self.data_mem[offset + i] = int(val) & 0xFF
    
    def launch_kernel(self, num_threads: int):
        self.threads = []
        num_blocks = (num_threads + self.threads_per_block - 1) // self.threads_per_block
        
        for block_id in range(num_blocks):
            for local_tid in range(self.threads_per_block):
                global_tid = block_id * self.threads_per_block + local_tid
                if global_tid < num_threads:
                    self.threads.append(ThreadState(
                        thread_id=local_tid,
                        block_id=block_id,
                        block_dim=self.threads_per_block,
                        registers=[0] * 14
                    ))
        print(f"Launched {len(self.threads)} threads in {num_blocks} blocks")
    
    def _decode(self, instr: int):
        opcode = (instr >> 12) & 0xF
        rd = (instr >> 8) & 0xF
        rs1 = (instr >> 4) & 0xF
        rs2 = instr & 0xF
        imm8 = instr & 0xFF
        return opcode, rd, rs1, rs2, imm8
    
    def _get_reg(self, thread: ThreadState, reg: int) -> int:
        if reg == SpecialReg.BLOCK_IDX: return thread.block_id
        elif reg == SpecialReg.BLOCK_DIM: return thread.block_dim
        elif reg == SpecialReg.THREAD_IDX: return thread.thread_id
        else: return thread.registers[reg]
    
    def _set_reg(self, thread: ThreadState, reg: int, val: int):
        if reg < 14:
            thread.registers[reg] = int(val) & 0xFFFFFFFF
    
    def _execute_thread(self, thread: ThreadState):
        if thread.done:
            return
        
        instr = self.program_mem[thread.pc]
        opcode, rd, rs1, rs2, imm8 = self._decode(instr)
        
        if opcode == Opcode.NOP:
            pass
        elif opcode == Opcode.ADD:
            self._set_reg(thread, rd, self._get_reg(thread, rs1) + self._get_reg(thread, rs2))
        elif opcode == Opcode.SUB:
            self._set_reg(thread, rd, self._get_reg(thread, rs1) - self._get_reg(thread, rs2))
        elif opcode == Opcode.MUL:
            self._set_reg(thread, rd, self._get_reg(thread, rs1) * self._get_reg(thread, rs2))
        elif opcode == Opcode.DIV:
            divisor = self._get_reg(thread, rs2)
            self._set_reg(thread, rd, self._get_reg(thread, rs1) // divisor if divisor else 0)
        elif opcode == Opcode.LDR:
            addr = self._get_reg(thread, rs1) & 0xFF
            self._set_reg(thread, rd, self.data_mem[addr])
        elif opcode == Opcode.STR:
            # rd=data, rs1=addr
            addr = self._get_reg(thread, rs1) & 0xFF
            val = self._get_reg(thread, rd)
            self.data_mem[addr] = int(val) & 0xFF
        elif opcode == Opcode.CONST:
            self._set_reg(thread, rd, imm8)
        elif opcode == Opcode.CMP:
            diff = self._get_reg(thread, rs1) - self._get_reg(thread, rs2)
            if diff < 0: thread.condition = 0b100
            elif diff == 0: thread.condition = 0b010
            else: thread.condition = 0b001
        elif opcode == Opcode.BRnzp:
            nzp = (instr >> 8) & 0b111
            offset = instr & 0xFF
            if offset & 0x80: offset = offset - 256
            if thread.condition & nzp:
                thread.pc += offset
                return
        elif opcode == Opcode.RET:
            thread.done = True
            return
        
        thread.pc += 1
    
    def run(self, max_cycles: int = 10000) -> int:
        self.cycles = 0
        while self.cycles < max_cycles:
            if all(t.done for t in self.threads):
                break
            for thread in self.threads:
                self._execute_thread(thread)
            self.cycles += 1
        return self.cycles
    
    def get_data(self, start: int, length: int) -> List[int]:
        return self.data_mem[start:start+length]

class TinyGPU:
    def __init__(self, num_cores: int = 2, threads_per_block: int = 4):
        self.sim = GPUSimulator(num_cores, threads_per_block)
        self.asm = Assembler()
    
    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        program = [
            self.asm.MUL('R0', '%blockIdx', '%blockDim'),
            self.asm.ADD('R0', 'R0', '%threadIdx'),  # i = global thread id
            self.asm.CONST('R1', 0),                 # baseA
            self.asm.CONST('R2', n),                 # baseB  
            self.asm.CONST('R3', 2 * n),             # baseC
            self.asm.ADD('R4', 'R1', 'R0'),          # addr_a = baseA + i
            self.asm.LDR('R4', 'R4'),                # a_val = mem[addr_a]
            self.asm.ADD('R5', 'R2', 'R0'),          # addr_b = baseB + i
            self.asm.LDR('R5', 'R5'),                # b_val = mem[addr_b]
            self.asm.ADD('R6', 'R4', 'R5'),          # sum = a_val + b_val
            self.asm.ADD('R7', 'R3', 'R0'),          # addr_c = baseC + i
            self.asm.STR('R7', 'R6'),                # mem[addr_c] = sum
            self.asm.RET()
        ]
        self.sim.load_program(program)
        self.sim.load_data([int(x) for x in a], offset=0)
        self.sim.load_data([int(x) for x in b], offset=n)
        self.sim.launch_kernel(n)
        cycles = self.sim.run()
        print(f"Vector add completed in {cycles} cycles")
        return np.array(self.sim.get_data(2 * n, n), dtype=np.uint8)
    
    def vector_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        program = [
            self.asm.MUL('R0', '%blockIdx', '%blockDim'),
            self.asm.ADD('R0', 'R0', '%threadIdx'),
            self.asm.CONST('R1', 0),
            self.asm.CONST('R2', n),
            self.asm.CONST('R3', 2 * n),
            self.asm.ADD('R4', 'R1', 'R0'),
            self.asm.LDR('R4', 'R4'),
            self.asm.ADD('R5', 'R2', 'R0'),
            self.asm.LDR('R5', 'R5'),
            self.asm.MUL('R6', 'R4', 'R5'),
            self.asm.ADD('R7', 'R3', 'R0'),
            self.asm.STR('R7', 'R6'),
            self.asm.RET()
        ]
        self.sim.load_program(program)
        self.sim.load_data([int(x) for x in a], offset=0)
        self.sim.load_data([int(x) for x in b], offset=n)
        self.sim.launch_kernel(n)
        cycles = self.sim.run()
        print(f"Vector mul completed in {cycles} cycles")
        return np.array(self.sim.get_data(2 * n, n), dtype=np.uint8)
    
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        baseA, baseB, baseC = 0, M * K, M * K + K * N
        
        program = [
            self.asm.MUL('R0', '%blockIdx', '%blockDim'),
            self.asm.ADD('R0', 'R0', '%threadIdx'),  # i
            self.asm.CONST('R1', 1),                 # inc
            self.asm.CONST('R2', N),                 # N
            self.asm.CONST('R3', baseA),
            self.asm.CONST('R4', baseB),
            self.asm.CONST('R5', baseC),
            self.asm.DIV('R6', 'R0', 'R2'),          # row = i / N
            self.asm.MUL('R7', 'R6', 'R2'),
            self.asm.SUB('R7', 'R0', 'R7'),          # col = i % N
            self.asm.CONST('R8', 0),                 # acc = 0
            self.asm.CONST('R9', 0),                 # k = 0
            # LOOP (offset 12):
            self.asm.MUL('R10', 'R6', 'R2'),         # row * N
            self.asm.ADD('R10', 'R10', 'R9'),        # + k
            self.asm.ADD('R10', 'R10', 'R3'),        # + baseA
            self.asm.LDR('R10', 'R10'),              # A[row,k]
            self.asm.MUL('R11', 'R9', 'R2'),         # k * N
            self.asm.ADD('R11', 'R11', 'R7'),        # + col
            self.asm.ADD('R11', 'R11', 'R4'),        # + baseB
            self.asm.LDR('R11', 'R11'),              # B[k,col]
            self.asm.MUL('R12', 'R10', 'R11'),       # A * B
            self.asm.ADD('R8', 'R8', 'R12'),         # acc += 
            self.asm.ADD('R9', 'R9', 'R1'),          # k++
            self.asm.CMP('R9', 'R2'),                # k < N?
            self.asm.BRn(-12),                       # loop
            self.asm.ADD('R9', 'R5', 'R0'),          # addr_c
            self.asm.STR('R9', 'R8'),                # store
            self.asm.RET()
        ]
        
        self.sim.load_program(program)
        self.sim.load_data([int(x) for x in A.flatten()], offset=baseA)
        self.sim.load_data([int(x) for x in B.flatten()], offset=baseB)
        self.sim.launch_kernel(M * N)
        cycles = self.sim.run()
        print(f"Matmul {M}x{K} @ {K}x{N} completed in {cycles} cycles")
        return np.array(self.sim.get_data(baseC, M * N), dtype=np.uint8).reshape(M, N)

if __name__ == "__main__":
    print("=" * 60)
    print("TINY GPU - Python Interface")
    print("=" * 60)
    
    gpu = TinyGPU(num_cores=2, threads_per_block=4)
    
    print("\n1. VECTOR ADDITION")
    a = np.array([1, 2, 3, 4], dtype=np.uint8)
    b = np.array([5, 6, 7, 8], dtype=np.uint8)
    c = gpu.vector_add(a, b)
    print(f"   {a} + {b} = {c}")
    expected = (a.astype(int) + b.astype(int)).astype(np.uint8)
    assert np.array_equal(c, expected), f"Failed! Expected {expected}"
    print("   PASS!")
    
    print("\n2. VECTOR MULTIPLICATION")
    c = gpu.vector_mul(a, b)
    print(f"   {a} * {b} = {c}")
    expected = (a.astype(int) * b.astype(int)).astype(np.uint8)
    assert np.array_equal(c, expected), f"Failed! Expected {expected}"
    print("   PASS!")
    
    print("\n3. MATRIX MULTIPLICATION")
    A = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    B = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    C = gpu.matmul(A, B)
    C_expected = (A.astype(int) @ B.astype(int)).astype(np.uint8)
    print(f"   A =\n{A}")
    print(f"   B =\n{B}")  
    print(f"   C = A @ B =\n{C}")
    print(f"   Expected =\n{C_expected}")
    assert np.array_equal(C, C_expected), f"Failed!"
    print("   PASS!")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
