//
// Created by andwh on 24/10/2024.
//

#ifndef SIMULATION_CUH
#define SIMULATION_CUH


#include "../automata_parser/uppaal_xml_parser.h"
#include "../automata_parser/abstract_parser.h"
#include "state/SharedModelState.cuh"
#include <cmath>
#include <cfloat>

#include "state/SharedModelState.cuh"
#include "state/SharedRunState.cuh"
#include "../main.cuh"

class simulation {
public:
    abstract_parser* parser;

    simulation(abstract_parser* uppaal_xml_parser_instance) { parser = uppaal_xml_parser_instance; }

    void run_statistical_model_checking(SharedModelState* model, float confidence, float precision, VariableKind* kinds, int num_vars);


};

double evaluate_expression_node_coalesced(const expr*, SharedBlockMemory*, double*, int);


#endif //SIMULATION_CUH
