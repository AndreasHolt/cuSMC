//
// Created by andwh on 04/11/2024.
//

#include "main.cuh"

#include <chrono>
#include "engine/Domain.h"
#include "automata_parser/uppaal_xml_parser.h"
#include <iostream>
#include "output.h"
#include "network_optimization/domain_optimization_visitor.h"
#include "network_optimization/pn_compile_visitor.h"
#include "simulation/simulation.cuh"
#include "simulation/simulation_config.h"
#include "simulation/state/SharedModelState.cuh"
#include "automata_parser/VariableUsageVisitor.h"

#include "simulation/simulation.cuh"

VariableKind* createKindArray(const std::unordered_map<int, VariableTrackingVisitor::VariableUsage>& registry) {
    VariableKind* kinds = new VariableKind[registry.size()];
    for(int i = 0; i < registry.size(); i++) {
        kinds[i] = registry.at(i).kind;
    }
    return kinds;
}


int main()
{
    // Hardcoded path to the XML file
    std::string filename = "../automata_parser/XmlFiles/UppaalBehaviorTest3.xml";
    string query1 = "c2.g3";
    string query2 = "c2.g4";
    std::unordered_set<std::string>* query_set = new std::unordered_set<std::string>();
    query_set->insert(query1);
    query_set->insert(query2);

    abstract_parser* parser = new uppaal_xml_parser();

    if constexpr (VERBOSE) {std::cout << "Test after instantiate_parser" << std::endl;}

    network model = parser->parse(filename);

    if constexpr (VERBOSE) {std::cout << "Parsing successful. Network details:" << std::endl;}

    if constexpr (VERBOSE) {cout << "Performing optimizations..." << endl;}

    network_props properties = {};
    simulation_config config = {};

    auto sim = simulation(parser);

    properties.node_edge_map = new std::unordered_map<int, std::list<edge>>(parser->get_node_edge_map());
    properties.start_nodes = new std::list<int>(parser->get_start_nodes());
    properties.template_names = new std::unordered_map<int, std::string>(*parser->get_template_names());
    properties.variable_names = new std::unordered_map<int, std::string>(*parser->get_clock_names());    // this can create mem leaks.
    properties.node_network = new std::unordered_map<int, int>(*parser->get_subsystems());
    properties.node_names = new std::unordered_map<int, std::string>(*parser->get_nodes_with_name());

    domain_optimization_visitor optimizer = domain_optimization_visitor(
        query_set,
        properties.node_network,
        properties.node_names,
        properties.template_names);
    optimizer.optimize(&model);

    pn_compile_visitor pn_compiler;
    pn_compiler.visit(&model);

    setup_simulation_config(&config, &model, optimizer.get_max_expr_depth(), optimizer.get_max_fanout(), optimizer.get_node_count());

    VariableTrackingVisitor var_tracker;
    var_tracker.visit(&model);
    if constexpr (VERBOSE) {
        var_tracker.print_variable_usage();
    }
    auto registry = var_tracker.get_variable_registry();

    VariableKind* kinds = createKindArray(registry);
    int num_vars = registry.size();

    if constexpr (VERBOSE) {
        for(int i = 0; i < registry.size(); i++) {
            printf("Kind %d: %d\n", i, kinds[i]);
        }
    }

    SharedModelState* state = init_shared_model_state(
    &model,  // cpu_network
    *optimizer.get_node_subsystems_map(),
    *properties.node_edge_map,
    optimizer.get_node_map(),
    var_tracker.get_variable_registry(),
    parser, num_vars
    );

    sim.run_statistical_model_checking(state, 0.05, 0.01, kinds, num_vars);


    // Kernels for debugging purposes
    if constexpr (VERBOSE) {
        verify_expressions_kernel<<<1,1>>>(state);
        // cudaDeviceSynchronize();
        // verify_invariants_kernel<<<1, 1>>>(state);
        // test_kernel<<<1, 1>>>(state);
        // validate_edge_indices<<<1, 1>>>(state);
    }

    delete parser;

    return 0;
}
