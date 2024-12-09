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

int main() {
    std::string filename = "../automata_parser/xml_files/UppaalBehaviorTest3.xml";

    // Regular queries
    // string query1 = "c1.f2";
    // string query2 = "c1.f4";
    std::unordered_set<std::string> *query_set = new std::unordered_set<std::string>();
    // query_set->insert(query1);
    // query_set->insert(query2);

    // Variable queries
    int variable_id = 5; // Val : -1 implies not variable checking query

    abstract_parser *parser = new uppaal_xml_parser();

    if constexpr (VERBOSE) { std::cout << "Test after instantiate_parser" << std::endl; }

    network model = parser->parse(filename);

    if constexpr (VERBOSE) { std::cout << "Parsing successful. Network details:" << std::endl; }

    if constexpr (VERBOSE) { cout << "Performing optimizations..." << endl; }

    network_props properties = {};
    populate_properties(properties, parser);

    // Optimize the model
    domain_optimization_visitor optimizer = domain_optimization_visitor(
        query_set,
        properties.node_network,
        properties.node_names,
        properties.template_names);
    optimizer.optimize(&model);

    // Compile expressions to PN
    pn_compile_visitor pn_compiler;
    pn_compiler.visit(&model);

    // Gather relevant information about variables
    VariableTrackingVisitor var_tracker;
    var_tracker.visit(&model);

    auto registry = var_tracker.get_variable_registry();

    VariableKind *kinds = var_tracker.createKindArray(registry);
    int num_vars = registry.size();

    // Statistics
    int simulations = 1000;
    bool isMax = true; // Gather info on either the max value of the variable or the min

    if (variable_id != -1) {
        std::unordered_map<int, node *> node_map = optimizer.get_node_map();
        SharedModelState *state = init_shared_model_state(
            &model, // cpu_network
            *optimizer.get_node_subsystems_map(),
            *properties.node_edge_map,
            node_map,
            var_tracker.get_variable_registry(),
            parser,
            num_vars);
        Statistics stats(simulations, VAR_STAT);

        printf("Running SMC\n");
        run_statistical_model_checking(state, 0.05, 0.01, kinds, num_vars, stats.get_comp_device_ptr(),
                                           stats.get_var_device_ptr(), variable_id, isMax, simulations);
        double *var_data = stats.collect_var_data();
        for (int i = 0; i < simulations; i++) {
            printf("Variable value for simulation %d is %f\n", i, var_data[i]);
        }
    }

    // Query analysis loop
    for (auto itr = query_set->cbegin(); itr != query_set->cend(); itr++) {
        Statistics stats(simulations, COMP_STAT);

        // Query string from set
        string query = *(*query_set).find(*itr);

        if constexpr (VERBOSE) {
            cout << "Recorded query is: " + query << endl;
        }

        // String split
        std::vector<char> component;
        std::vector<char> goal_node;
        bool period_reached = false;
        for (int i = 0; i < query.length(); i++) {
            // Guard
            if (query[i] == '.') {
                period_reached = true;
                continue;
            }
            if (!period_reached) {
                component.push_back(query[i]);
            }
            if (period_reached) {
                goal_node.push_back(query[i]);
            }
        }

        std::string component_name(component.begin(), component.end());
        std::string node_name(goal_node.begin(), goal_node.end());
        auto temp = properties.node_name_int_map.find(node_name);
        std::unordered_map<int, node *> node_map = optimizer.get_node_map();

        if (temp != properties.node_name_int_map.cend()) {
            int goal_node_idx = (*temp).second;

            (*node_map.find(goal_node_idx)).second->type = node::goal;
        }

        SharedModelState *state = init_shared_model_state(
            &model, // cpu_network
            *optimizer.get_node_subsystems_map(),
            *properties.node_edge_map,
            node_map,
            var_tracker.get_variable_registry(),
            parser,
            num_vars);

        // Run the SMC simulations
        run_statistical_model_checking(state, 0.05, 0.01, kinds, num_vars, stats.get_comp_device_ptr(),
                                           stats.get_var_device_ptr(), variable_id, isMax, simulations);
        try {
            auto results = stats.collect_results();
            stats.print_results(query, results);
        } catch (const std::runtime_error &e) {
            cout << "Error while collecting the results from the simulations: " << e.what() << endl;
            continue; // Decide whether to continue to the next query or to exit main
        }
    }

    // Kernels for debugging purposes
    if constexpr (VERBOSE) {
        // verify_expressions_kernel<<<1,1>>>(state);
        // test_kernel<<<1, 1>>>(state);
        // validate_edge_indices<<<1, 1>>>(state);
    }

    delete parser;

    return 1;
}
