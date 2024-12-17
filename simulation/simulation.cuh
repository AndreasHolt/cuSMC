//
// Created by andwh on 24/10/2024.
//

#ifndef SIMULATION_CUH
#define SIMULATION_CUH


#include "../automata_parser/uppaal_xml_parser.h"
#include "state/shared_model_state.cuh"
#include "state/shared_run_state.cuh"
#include "../main.cuh"
#include "statistics.cuh"


void run_statistical_model_checking(SharedModelState* model, float confidence, float precision, VariableKind* kinds, bool* flags, double* variable_flags, int variable_id);


__global__ void simulation_kernel(SharedModelState *model, bool *results,
                                  int runs_per_block, float time_bound, VariableKind *kinds, uint32_t  num_vars, bool* flags, double* variable_flags, int variable_id, bool isMax,
                                  curandState *rng_states_global, int curand_seed, int max_components, var_at_time *var_over_time, int *var_over_time_entries);


#endif //SIMULATION_CUH
