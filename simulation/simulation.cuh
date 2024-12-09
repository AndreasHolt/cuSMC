//
// Created by andwh on 24/10/2024.
//

#ifndef SIMULATION_CUH
#define SIMULATION_CUH


#include "../automata_parser/uppaal_xml_parser.h"
#include "state/shared_model_state.cuh"
#include "state/shared_run_state.cuh"

void run_statistical_model_checking(SharedModelState* model, float confidence, float precision, VariableKind* kinds, int num_vars, bool* flags, double* variable_flags, int variable_id, bool isMax, int num_simulations);
double evaluate_expression_node_coalesced(const expr*, SharedBlockMemory*, double*, int);


#endif //SIMULATION_CUH
