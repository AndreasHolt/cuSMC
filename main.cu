//
// Created by andwh on 04/11/2024.
//

#include "main.cuh"
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
#include "smc.cuh"



int main() {
    std::string filename = "../xml_files/UppaalBehaviorTest3.xml";

    int simulation_arr[] = {1000};

    for (int i : simulation_arr) {
        smc(filename, "", true, false, 0, 5, i);
    }


    return 1;
}
