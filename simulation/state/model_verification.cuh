//
// Created by andwh on 09/12/2024.
//

#ifndef MODEL_VERIFICATION_CUH
#define MODEL_VERIFICATION_CUH
#include "shared_model_state.cuh"


__global__ void test_kernel(SharedModelState* model);
__global__ void validate_edge_indices(SharedModelState* model);
__global__ void verify_expressions_kernel(SharedModelState* model);





#endif //MODEL_VERIFICATION_CUH
