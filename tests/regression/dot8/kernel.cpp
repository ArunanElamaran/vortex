#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	// New Code
    auto A = reinterpret_cast<int8_t*>(arg->A_addr);
	auto B = reinterpret_cast<int8_t*>(arg->B_addr);
	auto C = reinterpret_cast<int32_t*>(arg->C_addr);
    auto size = arg->size;

    int col = blockIdx.x;
    int row = blockIdx.y;

    TYPE sum(0);
    for (int e = 0; e < size; e+=4) {
        // Pack 4 int8_t elements from A and B into 32-bit integers
        uint32_t packedA = *((int*)(&A[row * size + e]));
        uint32_t packedB = ((uint8_t)B[(e+0)*size + col] << 0)
                         | ((uint8_t)B[(e+1)*size + col] << 8)
                         | ((uint8_t)B[(e+2)*size + col] << 16)
                         | ((uint8_t)B[(e+3)*size + col] << 24);
        // Accumulate the dot product result into the C
        sum += vx_dot8(packedA, packedB);
    }

    C[row * size + col] = sum;
    
    // // Original Code
    // auto A = reinterpret_cast<TYPE*>(arg->A_addr);
	// auto B = reinterpret_cast<TYPE*>(arg->B_addr);
	// auto C = reinterpret_cast<TYPE*>(arg->C_addr);
    // auto size = arg->size;

    // int col = blockIdx.x;
    // int row = blockIdx.y;

    // TYPE sum(0);
    // for (int e = 0; e < size; ++e) {
    //     sum += A[row * size + e] * B[e * size + col];
    // }

    // C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
