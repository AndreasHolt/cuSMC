//
// Created by andreas on 12/11/24.
//

#ifndef SMC_CUH
#define SMC_CUH

#include <chrono>
#include "include/engine/domain.h"
#include "automata_parser/uppaal_xml_parser.h"
#include <iostream>
#include "automata_parser/network/network_props.h"
#include "automata_parser/network/domain_optimization_visitor.h"
#include "automata_parser/network/pn_compile_visitor.h"
#include "simulation/simulation.cuh"
#include "simulation/state/shared_model_state.cuh"
#include "automata_parser/variable_usage_visitor.h"
#include "simulation/statistics.cuh"
#include "simulation/simulation.cuh"


void smc(configuration conf, statistics_Configuration stat_conf);




#endif //SMC_CUH
