#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>
#include <vx_print.h>

namespace vt = vortex::tensor;

// Software fp16 to fp32 conversion for verification
// IEEE 754 half-precision: 1 sign, 5 exponent, 10 mantissa
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

    // MMA; All data stays in tensor memory, only addresses in registers
    ctx::mma_sync(tmemC, tmemA, tmemB, tmemC);

  }

  // Store the computed C tile from tensor memory to global memory
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, tmemC, N);
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
