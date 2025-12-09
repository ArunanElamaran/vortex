# UMMA (Unified Matrix Multiply-Accumulate) Implementation

## Overview

UMMA is a new tensor instruction for Vortex that uses **tensor memory (TMEM)** to store matrix tiles, with registers holding only **addresses** pointing to the data. This differs from WMMA where registers hold the actual matrix data.

## Key Differences: WMMA vs UMMA

| Aspect | WMMA | UMMA |
|--------|------|------|
| Data storage | Floating-point registers (f0-f31) | Tensor Memory (TMEM) |
| Register contents | Actual matrix elements | Addresses to TMEM |
| Register type | Float registers | Integer registers (a0, a1, a2) |
| Instruction funct3 | 0 | 1 |

## Instruction Encoding

Both WMMA and UMMA use RISC-V CUSTOM0 opcode (0x0b) with R-type format:

```
| funct7 (7 bits) | rs2 (5 bits) | rs1 (5 bits) | funct3 (3 bits) | rd (5 bits) | opcode (7 bits) |
|      2          |      0       |   fmt_s      |       1         |    fmt_d    |      0x0b       |
```

- **funct7 = 2**: Tensor core operation
- **funct3 = 1**: UMMA (vs 0 for WMMA)
- **rd**: Output format ID (e.g., 8 for int32)
- **rs1**: Input format ID (e.g., 9 for int8)
- **Addresses**: Passed in fixed registers a0 (A tile), a1 (B tile), a2 (C/D tile)

## Files Modified

### 1. `/vortex/kernel/include/vx_tensor.h`

Added `umma_context` template class alongside existing `wmma_context`:

```cpp
template <uint32_t NT, typename It, typename Ot>
struct umma_context {
  // Fragment types that hold TMEM addresses instead of data
  template <frag_use_t Use>
  struct fragment_addr_t {
    uintptr_t addr;  // 32-bit on RV32
  };

  using fragment_a   = fragment_addr_t<matrix_a>;
  using fragment_b   = fragment_addr_t<matrix_b>;
  using fragment_acc = fragment_addr_t<accumulator>;

  // Load from global memory to TMEM
  static void load_matrix_sync(fragment_addr &frag, const void *src, size_t ldm);
  
  // Store from TMEM to global memory
  static void store_matrix_sync(void *dst, const fragment_addr &frag, size_t ldm);
  
  // Fill TMEM region with value
  static void fill_fragment(fragment_addr &frag, output_t value);
  
  // Execute UMMA instruction
  static void mma_sync(fragment_acc &fragD, const fragment_a &fragA,
                       const fragment_b &fragB, const fragment_acc &fragC);
};
```

Key changes:
- `wmma_config_t` now receives `It, Ot` template parameters for correct `i_ratio` calculation
- Tile dimensions: `tileK = xtileK * i_ratio` (e.g., for int8: tileK = 4 * 4 = 16)
- Fragment addresses use `uintptr_t` (not `uint64_t`) for RV32 compatibility
- TMEM pointers are marked `volatile` to prevent optimization issues

### 2. `/vortex/sim/simx/decode.cpp`

Added UMMA decoding in the TCU (Tensor Core Unit) case:

```cpp
case 1: { // UMMA - funct3 = 1
  uint32_t fmt_d = rd;   // Output format from rd field
  uint32_t fmt_s = rs1;  // Input format from rs1 field
  
  // Fixed registers for addresses (by convention)
  constexpr uint32_t reg_addr_a = 10;  // a0
  constexpr uint32_t reg_addr_b = 11;  // a1
  constexpr uint32_t reg_addr_c = 12;  // a2
  
  // Generate micro-ops for tile steps
  for (k, m, n steps...) {
    instr->setOpType(TcuType::UMMA);
    instr->setArgs(IntrTcuArgs{fmt_s, fmt_d, m, n});
    instr->setDestReg(reg_addr_c, RegType::Integer);
    instr->setSrcReg(0, reg_addr_a, RegType::Integer);
    instr->setSrcReg(1, reg_addr_b, RegType::Integer);
    instr->setSrcReg(2, reg_addr_c, RegType::Integer);
  }
}
```

### 3. `/vortex/sim/simx/tensor_unit.cpp`

Added `umma()` function to execute the UMMA instruction:

```cpp
void umma(uint32_t wid, uint32_t fmt_s, uint32_t fmt_d,
          uint32_t step_m, uint32_t step_n,
          const std::vector<reg_data_t>& rs1_data,  // A address
          const std::vector<reg_data_t>& rs2_data,  // B address
          const std::vector<reg_data_t>& rs3_data,  // C address
          std::vector<reg_data_t>& rd_data,
          ExeTraceData* trace_data) {
  
  // Only execute on step (0,0) - process full tile at once
  if (step_m != 0 || step_n != 0) {
    rd_data[0].u32 = rs3_data[0].u32;
    return;
  }
  
  // Get addresses from integer registers (32-bit)
  uint64_t addr_a = rs1_data[0].u32;
  uint64_t addr_b = rs2_data[0].u32;
  uint64_t addr_c = rs3_data[0].u32;
  
  // Calculate tile sizes based on format IDs
  uint32_t tileM = cfg::xtileM;
  uint32_t tileN = cfg::xtileN;
  uint32_t tileK = cfg::xtileK * i_ratio;
  
  // Read full tiles from tensor memory
  tensor_mem->read(a_bytes.data(), addr_a, a_tile_bytes);
  tensor_mem->read(b_bytes.data(), addr_b, b_tile_bytes);
  tensor_mem->read(c_bytes.data(), addr_c, c_tile_bytes);
  
  // Perform MMA: D = A * B + C
  for (m, n, k...) {
    acc += a_val * b_val;
  }
  
  // Write result back to tensor memory
  tensor_mem->write(d_bytes.data(), addr_c, c_tile_bytes);
}
```

### 4. `/vortex/sim/simx/tensor_mem.cpp`

Enhanced tensor memory to properly handle TMEM address detection:

```cpp
bool TensorMem::is_tensor_mem_addr(uint64_t addr) const {
  // Check if address falls within TMEM range
  return (addr >= TMEM_BASE_ADDR) && (addr < TMEM_BASE_ADDR + TMEM_SIZE);
}

uint64_t TensorMem::to_local_addr(uint64_t global_addr) const {
  // Convert global TMEM address to local offset
  return global_addr - TMEM_BASE_ADDR;
}
```

### 5. `/vortex/sim/simx/types.h`

Added `UMMA` to the `TcuType` enum:

```cpp
enum class TcuType {
  WMMA,
  UMMA,  // New
};
```

## Tile Dimensions

For NT=4 threads with int8 input and int32 output:

| Parameter | Value | Calculation |
|-----------|-------|-------------|
| xtileM | 4 | Based on NT and NR |
| xtileN | 2 | Based on NT and NR |
| xtileK | 4 | tile_cap / max(xtileM, xtileN) |
| i_ratio | 4 | sizeof(float) / sizeof(int8_t) |
| **tileM** | 8 | xtileM (for tile addressing) |
| **tileN** | 4 | xtileN (for tile addressing) |
| **tileK** | 16 | xtileK * i_ratio |

Tile sizes in bytes:
- A tile: tileM × tileK × sizeof(int8) = 8 × 16 × 1 = 128 bytes
- B tile: tileK × tileN × sizeof(int8) = 16 × 4 × 1 = 64 bytes
- C tile: tileM × tileN × sizeof(int32) = 8 × 4 × 4 = 128 bytes
- **Total per block**: 320 bytes

## Tensor Memory Layout

```
TMEM_BASE_ADDR (0xFFFF4000)
├── Block 0
│   ├── A tile (128 bytes) @ offset 0x00
│   ├── B tile (64 bytes)  @ offset 0x80
│   └── C tile (128 bytes) @ offset 0xC0
├── Block 1
│   ├── A tile @ offset 0x140
│   ├── B tile @ offset 0x1C0
│   └── C tile @ offset 0x200
└── ...
```

## Usage Example (Kernel Code)

```cpp
using ctx = vt::umma_context<NUM_THREADS, vt::int8, vt::int32>;

ctx::fragment_a   fragA;
ctx::fragment_b   fragB;
ctx::fragment_acc fragC;

// Set up TMEM addresses
uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
uint64_t block_base = TMEM_BASE_ADDR + block_id * (a_size + b_size + c_size);

ctx::set_fragment_addr(fragA, (void*)(block_base));
ctx::set_fragment_addr(fragB, (void*)(block_base + a_size));
ctx::set_fragment_addr(fragC, (void*)(block_base + a_size + b_size));

// Initialize accumulator
ctx::fill_fragment(fragC, 0);

// Load and compute
for (uint32_t i = 0; i < K; i += ctx::tileK) {
    ctx::load_matrix_sync(fragA, pA + tile_row * K + i, K);
    ctx::load_matrix_sync(fragB, pB + i * N + tile_col, N);
    ctx::mma_sync(fragC, fragA, fragB, fragC);
}

// Store result
ctx::store_matrix_sync(pC + tile_row * N + tile_col, fragC, N);
```

## Current Status

### Completed
- [x] UMMA instruction encoding (funct3=1, funct7=2)
- [x] Decoder support with fixed address registers (a0, a1, a2)
- [x] Tensor memory read/write with address translation
- [x] umma_context template with proper tile dimensions
- [x] RV32-compatible addressing (uintptr_t instead of uint64_t)

### In Progress
- [ ] Fixing instruction encoding in inline assembly
- [ ] Debugging memory access patterns
- [ ] End-to-end test validation

### Known Issues
1. The inline assembly `.word %[insn]` approach for encoding format IDs needs verification
2. Memory access violation when storing results to global memory (investigating C buffer allocation flags)
