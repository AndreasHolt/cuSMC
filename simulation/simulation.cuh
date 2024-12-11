//
// Created by andwh on 24/10/2024.
//

#ifndef SIMULATION_CUH
#define SIMULATION_CUH


#include "../automata_parser/uppaal_xml_parser.h"
#include "state/shared_model_state.cuh"
#include "state/shared_run_state.cuh"
#include "../main.cuh"

void run_statistical_model_checking(SharedModelState* model, float confidence, float precision, VariableKind* kinds, bool* flags, double* variable_flags, int variable_id, configuration conf, model_info m_info);

#endif //SIMULATION_CUH
