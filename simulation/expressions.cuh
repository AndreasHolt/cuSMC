//
// Created by andwh on 10/12/2024.
//

#ifndef EXPRESSIONS_CUH
#define EXPRESSIONS_CUH
#include "state/shared_run_state.cuh"

extern __device__ double evaluate_expression(const expr *e, SharedBlockMemory *shared);

#endif //EXPRESSIONS_CUH
