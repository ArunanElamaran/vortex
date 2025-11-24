#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  // Calculate unique accumulator address in tensor memory for this block
  // Each block gets its own space in tensor memory
  uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
  uint32_t accum_size = sizeof(ctx::output_t) * ctx::tileM * ctx::tileN;
  auto pAccum = reinterpret_cast<ctx::output_t*>(TMEM_ACCUM_BASE + block_id * accum_size);

  // Initialize accumulator tile to zero
  ctx::fill_fragment(fragC, 0);
  
  // Store initial zero accumulator to tensor memory
  ctx::store_matrix_sync(pAccum, fragC, ctx::tileN);

  for (int i = 0; i < K; i += ctx::tileK) {
    // Load accumulator from tensor memory at the start of each iteration
    ctx::load_matrix_sync(fragC, pAccum, ctx::tileN);
    
    auto pTileA = pA + tile_row * K + i;

    // Load A tile
    ctx::load_matrix_sync(fragA, pTileA, K);

    // Load B tile
    if constexpr (vt::ITYPE::bits < 8) {
      // For sub-byte matrix B must be in col-major format
      auto pTileB = pB + tile_col * K + i;
      ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    } else {
      auto pTileB = pB + i * N + tile_col;
      ctx::load_matrix_sync(fragB, pTileB, N);
    }

    // Matrix multiply-accumulate: c += a * b
    ctx::mma_sync(fragC, fragA, fragB, fragC);
    
    // Store accumulator back to tensor memory after each iteration
    ctx::store_matrix_sync(pAccum, fragC, ctx::tileN);
  }

  // Load final accumulator from tensor memory
  ctx::load_matrix_sync(fragC, pAccum, ctx::tileN);
  
  // Store the computed C tile to global memory
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
