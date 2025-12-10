# UMMA (Unified Matrix Multiply-Accumulate) Implementation

## Overview

UMMA is a tensor unit instruction that performs matrix multiplication and accumulation using **tensor memory (TMEM)** instead of register files. Unlike WMMA (which uses register-packed data), UMMA operates on contiguous memory regions in TMEM, simplifying the programming model while leveraging specialized tensor memory.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Kernel Code                          │
│  ctx::mma_sync(tmemC, tmemA, tmemB, tmemC)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    UMMA Instruction                         │
│  Registers contain TMEM addresses: a0=A, a1=B, a2=C        │
│  Format encoded in instruction: fmt_s (input), fmt_d (out) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Tensor Unit (tensor_unit.cpp)                 │
│  1. Read tiles from TMEM                                    │
│  2. Compute D = C + A × B                                   │
│  3. Write result back to TMEM                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Tensor Memory (tensor_mem.cpp)                │
│  Per-core scratchpad memory for tensor operations           │
│  Direct read/write access (no cache hierarchy)              │
└─────────────────────────────────────────────────────────────┘
```

## Tile Dimensions

The UMMA tile dimensions are derived from the base configuration and input type:

| Parameter | Formula | Example (fp16→fp32) |
|-----------|---------|---------------------|
| `tileM` | `xtileM` | 8 |
| `tileN` | `xtileN` | 4 |
| `tileK` | `xtileK × i_ratio` | 4 × 2 = 8 |

Where:
- `xtileM`, `xtileN`, `xtileK` are base tile dimensions (8, 4, 4 for 4 threads)
- `i_ratio = 4 / sizeof(input_type)` (e.g., 2 for fp16, 4 for int8)

## Memory Layout

All tiles are stored in **row-major** order in tensor memory:

```
Tensor Memory Layout:
┌────────────────────────────────────────┐
│ A tile: tileM × tileK elements         │
│ [a00, a01, ..., a0K, a10, a11, ...]   │
├────────────────────────────────────────┤
│ B tile: tileK × tileN elements         │
│ [b00, b01, ..., b0N, b10, b11, ...]   │
├────────────────────────────────────────┤
│ C tile: tileM × tileN elements         │
│ [c00, c01, ..., c0N, c10, c11, ...]   │
└────────────────────────────────────────┘
```

## Implementation Details

### 1. Instruction Decoding (`decode.cpp`)

The UMMA instruction is encoded as:
```
R-type format: funct7[6:0] | rs2[4:0] | rs1[4:0] | funct3[2:0] | rd[4:0] | opcode[6:0]
- opcode = RISCV_CUSTOM0 (0x0b)
- funct3 = 1 (distinguishes from WMMA which uses funct3=0)
- funct7 = 2
- rd = output format ID (fmt_d)
- rs1 = input format ID (fmt_s)
```

### 2. Execution Flow (`execute.cpp` → `tensor_unit.cpp`)

```cpp
case TcuType::UMMA: {
    tensor_unit_->umma(wid, fmt_s, fmt_d, step_m, step_n,
                       rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}
```

### 3. UMMA Function (`tensor_unit.cpp`)

The `umma()` function:

1. **Extract addresses** from registers (thread 0's values):
   ```cpp
   uint64_t addr_a = rs1_data[0].u32;  // A tile address in TMEM
   uint64_t addr_b = rs2_data[0].u32;  // B tile address in TMEM
   uint64_t addr_c = rs3_data[0].u32;  // C tile address in TMEM
   ```

2. **Calculate tile dimensions** based on format:
   ```cpp
   uint32_t input_bytes = get_type_bytes(fmt_s);
   uint32_t i_ratio = 4 / input_bytes;
   uint32_t tileK = xtileK * i_ratio;
   ```

3. **Dispatch to appropriate executor**:
   ```cpp
   auto umma_fn = select_UMMA(fmt_s, fmt_d);
   umma_fn(tensor_mem, addr_a, addr_b, addr_c, xtileM, xtileN, tileK);
   ```

4. **Return address to ALL threads**:
   ```cpp
   for (size_t t = 0; t < rd_data.size(); ++t) {
       rd_data[t].u32 = static_cast<uint32_t>(addr_c);
   }
   ```

### 4. Matrix Computation (`execute_umma<It, Ot>`)

```cpp
template <typename It, typename Ot>
static void execute_umma(const std::shared_ptr<TensorMem>& tensor_mem,
                         uint64_t addr_a, uint64_t addr_b, uint64_t addr_c,
                         uint32_t tileM, uint32_t tileN, uint32_t tileK) {
    // Read tiles from TMEM
    tensor_mem->read(a_tile.data(), addr_a, a_bytes);
    tensor_mem->read(b_tile.data(), addr_b, b_bytes);
    tensor_mem->read(c_tile.data(), addr_c, c_bytes);

    // Matrix multiply: D[m][n] = C[m][n] + sum_k(A[m][k] * B[k][n])
    for (uint32_t m = 0; m < tileM; ++m) {
        for (uint32_t n = 0; n < tileN; ++n) {
            otype acc = c_tile[m * tileN + n];
            for (uint32_t k = 0; k < tileK; ++k) {
                acc = FMA<It, Ot>::eval(a_tile[m * tileK + k],
                                        b_tile[k * tileN + n], acc);
            }
            d_tile[m * tileN + n] = acc;
        }
    }

    // Write result back to TMEM
    tensor_mem->write(d_tile.data(), addr_c, c_bytes);
}
```

## Supported Format Combinations

| Input Type (`fmt_s`) | Output Type (`fmt_d`) | `i_ratio` | `tileK` |
|---------------------|----------------------|-----------|---------|
| fp16 (id=1) | fp32 (id=0) | 2 | 8 |
| bf16 (id=2) | fp32 (id=0) | 2 | 8 |
| tf32 (id=3) | fp32 (id=0) | 1 | 4 |
| fp16 (id=1) | fp16 (id=1) | 2 | 8 |
| bf16 (id=2) | bf16 (id=2) | 2 | 8 |
| int8 (id=9) | int32 (id=8) | 4 | 16 |
| uint8 (id=10) | int32 (id=8) | 4 | 16 |

## FMA (Fused Multiply-Add) Templates

Each format combination has a specialized FMA template for proper type conversion:

```cpp
// Example: fp16 → fp32
template <>
struct FMA<vt::fp16, vt::fp32> {
    static float eval(uint16_t a, uint16_t b, float c) {
        auto xa = rv_htof_s(a, 0, nullptr);  // Convert fp16 to fp32
        auto xb = rv_htof_s(b, 0, nullptr);
        auto xab = rv_fmul_s(xa, xb, 0, nullptr);  // Multiply in fp32
        auto xc = bit_cast<uint32_t>(c);
        auto xd = rv_fadd_s(xab, xc, 0, nullptr);  // Add accumulator
        return bit_cast<float>(xd);
    }
};
```

## Tensor Memory Addressing

TMEM uses a simple linear addressing scheme:

```cpp
uint64_t to_local_addr(uint64_t addr) {
    uint64_t local = addr - TMEM_BASE_ADDR;
    return local & (config_.capacity - 1);  // Wrap within capacity
}
```

- `TMEM_BASE_ADDR` = 0xC0000000 (3GB mark)
- Capacity = 64KB per core (configurable via `TMEM_LOG_SIZE`)

## Key Differences: UMMA vs WMMA

| Aspect | WMMA | UMMA |
|--------|------|------|
| Data Location | Register file | Tensor memory |
| Data Layout | Packed in registers (complex) | Contiguous row-major (simple) |
| Addressing | Register indices | Memory addresses |
| Steps | Multiple (step_m, step_n) | Single instruction |
| Memory Access | Via loads/stores | Direct TMEM read/write |

## Critical Implementation Notes

### 1. All Threads Need Return Value

The UMMA instruction returns the output address, and **all threads** must receive this value:

```cpp
// CRITICAL: Set rd for ALL threads, not just thread 0
for (size_t t = 0; t < rd_data.size(); ++t) {
    rd_data[t].u32 = static_cast<uint32_t>(addr_c);
}
```

This is because `store_matrix_sync` uses this address, and each thread reads from its own register.

### 2. Per-Core TMEM

Tensor memory is **per-core**, not shared across cores. Each core has its own 64KB TMEM starting at the same base address (0xC0000000). The kernel must use warp-local offsets:

```cpp
uint32_t warp_id = vx_warp_id();
uint64_t block_tmem_base = TMEM_BASE_ADDR + warp_id * tile_size;
```

### 3. Type-Specific tileK

The K dimension varies based on input type due to packing:
- fp32/tf32: `tileK = 4`
- fp16/bf16: `tileK = 8`
- int8/uint8: `tileK = 16`

This is automatically calculated using `i_ratio = 4 / sizeof(input_type)`.

## File References

- **Instruction Decode**: `sim/simx/decode.cpp`
- **Instruction Execute**: `sim/simx/execute.cpp`
- **Tensor Unit**: `sim/simx/tensor_unit.cpp`
- **Tensor Memory**: `sim/simx/tensor_mem.cpp`
- **Type Definitions**: `sim/common/tensor_cfg.h`
- **Kernel API**: `kernel/include/vx_tensor.h`
