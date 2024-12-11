//
// Created by andwh on 24/10/2024.
//

#ifndef SIMULATION_CUH
#define SIMULATION_CUH


#include "../automata_parser/uppaal_xml_parser.h"
#include "state/shared_model_state.cuh"
#include "state/shared_run_state.cuh"

constexpr bool USE_GLOBAL_MEMORY_CURAND = true;

__global__ void simulation_kernel(SharedModelState *model, bool *results,
                                  int runs_per_block, float time_bound, VariableKind *kinds, int num_vars, bool* flags, double* variable_flags, int variable_id, bool isMax,
                                  curandState *rng_states_global);

#endif //SIMULATION_CUH
