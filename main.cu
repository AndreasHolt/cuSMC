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
#define VERBOSE 1
#include "simulation/simulation.cuh"

int main()
{
    // Hardcoded path to the XML file
    std::string filename = "../automata_parser/XmlFiles/UppaalBehaviorTest2.xml";
    string query1 = "c2.g3";
    string query2 = "c2.g4";
    std::unordered_set<std::string>* query_set = new std::unordered_set<std::string>();
    query_set->insert(query1);
    query_set->insert(query2);

    abstract_parser* parser = new uppaal_xml_parser();

    network model = parser->parse(filename);
    std::cout << "Parsing successful. Network details:" << std::endl;

    cout << "Performing optimizations..." << endl;


    network_props properties = {};
    simulation_config config = {};
    // properties.node_edge_map = new std::unordered_map<int, std::list<edge>>(*parser->get_node_edge_map());

    auto sim = simulation(parser);
    sim.runSimulation();

    cout << "test" << endl;
    // net.print_automatas();
    // Optimize by making static size arrays for following data.
    properties.node_edge_map = new std::unordered_map<int, std::list<edge>>(parser->get_node_edge_map());
    properties.start_nodes = new std::list<int>(parser->get_start_nodes());
    properties.template_names = new std::unordered_map<int, std::string>(*parser->get_template_names());
    properties.variable_names = new std::unordered_map<int, std::string>(*parser->get_clock_names());    // this can create mem leaks.
    properties.node_network = new std::unordered_map<int, int>(*parser->get_subsystems());
    // properties.clock_names = new std::unordered_map<int, std::string>(*parser->get_nodes_with_name());
    properties.node_names = new std::unordered_map<int, std::string>(*parser->get_nodes_with_name());
    // properties.clock_names = new std::unordered_map<int, std::string>(*parser->get_nodes_with_name());

    cout << "test" << endl;

    domain_optimization_visitor optimizer = domain_optimization_visitor(
        query_set,
        properties.node_network,
        properties.node_names,
        properties.template_names);
    optimizer.optimize(&model);

    pn_compile_visitor pn_compiler;
    pn_compiler.visit(&model);

    setup_simulation_config(&config, &model, optimizer.get_max_expr_depth(), optimizer.get_max_fanout(), optimizer.get_node_count());


    // SharedModelState* state = init_shared_model_state(&model, *optimizer.get_node_subsystems_map(), optimizer.get_node_map());
    SharedModelState* state = init_shared_model_state(&model, *optimizer.get_node_subsystems_map(), *properties.node_edge_map, optimizer.get_node_map());
    cout << "test" << endl;

    sim.run_statistical_model_checking(state, 0.05, 0.01);

    if (VERBOSE) {
        verify_invariants_kernel<<<1, 1>>>(state);
        test_kernel<<<1, 1>>>(state);
        validate_edge_indices<<<1, 1>>>(state);
    }



    // properties.node_network = new std::unordered_map<int, int>(*parser->get_subsystems());

    // We need a vars_list_: This has all the vars, their ids and their types.
    // Then we need node_names, to get the name of a node, i.e. 'f2' from the id
    // We also need the start node, to know which nodes a for different automatas/configurations

    // properties.pre_optimisation_start = std::chrono::steady_clock::now();
    delete parser;

    std::cout << "TEST";

    return 0;
}
