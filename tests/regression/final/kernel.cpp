#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>
#include <vx_print.h>

namespace vt = vortex::tensor;

// Software fp16 to fp32 conversion for verification
// IEEE 754 half-precision: 1 sign, 5 exponent, 10 mantissa
inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      // Zero
      f = sign << 31;
    } else {
      // Subnormal - convert to normalized fp32
      exp = 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FF;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    // Inf or NaN
    f = (sign << 31) | 0x7F800000 | (mant << 13);
  } else {
    // Normal number
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  
  float result;
  __builtin_memcpy(&result, &f, sizeof(result));
  return result;
}

// Use UMMA context - registers hold addresses to tensor memory
using ctx = vt::umma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  // Fragments simply contain tensor memory addresses
  ctx::fragment_a   tmemA;
  ctx::fragment_b   tmemB;
  ctx::fragment_acc tmemC;

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  // Calculate unique tensor memory addresses for this block's tiles
  uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
  
  // Layout in tensor memory:
  // - A tiles: starts at TMEM_BASE_ADDR
  // - B tiles: after A tiles
  // - C/D accumulators: after B tiles
  uint32_t a_tile_size = sizeof(ctx::input_t) * ctx::tileM * ctx::tileK;
  uint32_t b_tile_size = sizeof(ctx::input_t) * ctx::tileK * ctx::tileN;
  uint32_t c_tile_size = sizeof(ctx::output_t) * ctx::tileM * ctx::tileN;
  
  // Each block gets its own region in tensor memory
  uint64_t block_tmem_base = TMEM_ACCUM_BASE + block_id * (a_tile_size + b_tile_size + c_tile_size);
  
  // Set fragment addresses to point to tensor memory locations
  ctx::set_fragment_addr(tmemA, reinterpret_cast<void*>(block_tmem_base));
  ctx::set_fragment_addr(tmemB, reinterpret_cast<void*>(block_tmem_base + a_tile_size));
  ctx::set_fragment_addr(tmemC, reinterpret_cast<void*>(block_tmem_base + a_tile_size + b_tile_size));

  // Initialize accumulator in tensor memory to zero
  ctx::fill_fragment(tmemC, 0);

  // Counters for MMA verification
  uint32_t total_checks = 0;
  uint32_t total_mismatches = 0;

  for (uint32_t i = 0; i < K; i += ctx::tileK) {
    auto pTileA = pA + tile_row * K + i;

    // Load A tile from global memory to tensor memory
    ctx::load_matrix_sync(tmemA, pTileA, K);

    // Load B tile
    if constexpr (vt::ITYPE::bits < 8) {
      // For sub-byte matrix B must be in col-major format
      auto pTileB = pB + tile_col * K + i;
      ctx::load_matrix_sync<vt::col_major>(tmemB, pTileB, K);
    } else {
      auto pTileB = pB + i * N + tile_col;
      ctx::load_matrix_sync(tmemB, pTileB, N);
    }

    // Software reference MMA: compute expected result before hardware MMA
    // Thread 0 computes the full reference result for comparison
    // Store a copy of C before mma_sync, then compute reference, then compare
    ctx::output_t ref_C[ctx::tileM * ctx::tileN];
    if (vx_thread_id() == 0) {
      auto tmem_A = reinterpret_cast<volatile ctx::input_t*>(tmemA.addr);
      auto tmem_B = reinterpret_cast<volatile ctx::input_t*>(tmemB.addr);
      auto tmem_C = reinterpret_cast<volatile ctx::output_t*>(tmemC.addr);
      
      // Copy current C accumulator
      for (uint32_t idx = 0; idx < ctx::tileM * ctx::tileN; ++idx) {
        ref_C[idx] = tmem_C[idx];
      }
      
      // Compute reference: C[m][n] += A[m][k] * B[k][n]
      // A is tileM x tileK (row-major in tmem)
      // B is tileK x tileN (row-major in tmem)
      // C is tileM x tileN (row-major in tmem)
      for (uint32_t m = 0; m < ctx::tileM; ++m) {
        for (uint32_t n = 0; n < ctx::tileN; ++n) {
          ctx::output_t sum = ref_C[m * ctx::tileN + n];
          for (uint32_t k = 0; k < ctx::tileK; ++k) {
            ctx::input_t a_bits = tmem_A[m * ctx::tileK + k];
            ctx::input_t b_bits = tmem_B[k * ctx::tileN + n];
            // Properly convert fp16 bit patterns to fp32 values
            float a_val = fp16_to_fp32(a_bits);
            float b_val = fp16_to_fp32(b_bits);
            sum += a_val * b_val;
          }
          ref_C[m * ctx::tileN + n] = sum;
        }
      }
    }

    // MMA; All data stays in tensor memory, only addresses in registers
    ctx::mma_sync(tmemC, tmemA, tmemB, tmemC);

    // Compare hardware result with software reference
    if (vx_thread_id() == 0) {
      auto tmem_C = reinterpret_cast<volatile ctx::output_t*>(tmemC.addr);
      uint32_t iter_mismatches = 0;
      for (uint32_t m = 0; m < ctx::tileM; ++m) {
        for (uint32_t n = 0; n < ctx::tileN; ++n) {
          uint32_t idx = m * ctx::tileN + n;
          ctx::output_t hw_val = tmem_C[idx];
          ctx::output_t sw_val = ref_C[idx];
          // Use bit comparison for floats to detect any difference including NaN
          uint32_t hw_bits, sw_bits;
          __builtin_memcpy(&hw_bits, &hw_val, sizeof(hw_bits));
          __builtin_memcpy(&sw_bits, &sw_val, sizeof(sw_bits));
          total_checks++;
          if (hw_bits != sw_bits) {
            vx_printf("MMA MISMATCH C[%u,%u]: hw=0x%x sw=0x%x (block=%u,%u iter=%u)\n",
                     m, n, hw_bits, sw_bits, blockIdx.x, blockIdx.y, i);
            iter_mismatches++;
            total_mismatches++;
          }
        }
      }
      if (iter_mismatches == 0) {
        vx_printf("MMA verified OK (block=%u,%u iter=%u)\n", blockIdx.x, blockIdx.y, i);
      }
    }
  }

  // Print summary
  if (vx_thread_id() == 0) {
    vx_printf("MMA SUMMARY (block=%u,%u): %u mismatches / %u checks\n",
             blockIdx.x, blockIdx.y, total_mismatches, total_checks);
  }

  // Store the computed C tile from tensor memory to global memory
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, tmemC, N);
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
