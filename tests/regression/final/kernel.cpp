#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::umma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  // Global memory addresses of data
  auto pA = reinterpret_cast<const ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<const ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  // Calculate tile sizes
  uint32_t a_tile_size = sizeof(ctx::input_t) * ctx::tileM * ctx::tileK;
  uint32_t b_tile_size = sizeof(ctx::input_t) * ctx::tileK * ctx::tileN;
  uint32_t c_tile_size = sizeof(ctx::output_t) * ctx::tileM * ctx::tileN;

  // Allocate tensor memory for the tile of matrix A & C
	auto tensor_ptr = __tensor_mem(a_tile_size+c_tile_size);
  auto tensor_A = reinterpret_cast<ctx::input_t *>(tensor_ptr);;
  auto tensor_C = reinterpret_cast<ctx::output_t *>(reinterpret_cast<uint8_t*>(tensor_ptr) + a_tile_size);

  // Allocate local memory for the tile of matrix B
	auto local_ptr = __local_mem(b_tile_size);
  auto local_B = reinterpret_cast<ctx::input_t *>(local_ptr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    const auto* A_tile_global = pA + tile_row * K + k;
    const auto* B_tile_global = pB + k * N      + tile_col;
    auto*       C_tile_global = pC + tile_row * N + tile_col;

    // 1) Load tiles from global → local
    ctx::load_tile_sync_A(tensor_A, A_tile_global, K);
    ctx::load_tile_sync_B(local_B, B_tile_global, N);
    ctx::load_tile_sync_C(tensor_C, C_tile_global, N);

    // 2) Do UMMA in local memory
    ctx::umma_sync(tensor_C, tensor_A, local_B);

    // 3) Store result back to global
    ctx::store_tile_sync_C(C_tile_global, tensor_C, N);
  }
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
