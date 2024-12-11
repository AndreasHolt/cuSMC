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

void sim(string filename, string query, bool isMax, bool isEstimate, int variable_threshhold, int variable_id, int simulations) {
    // Read and parse XML file
    abstract_parser *parser = new uppaal_xml_parser();
    network model = parser->parse(filename);
    network_props properties = {};
    populate_properties(properties, parser);

    std::unordered_set<std::string> *query_set = new std::unordered_set<std::string>();
    query_set->insert(query);
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

    double result = 0;
    // Handling variable queries
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

        int len_of_array = simulations;
        double *var_data = stats.collect_var_data();
        // Estimate query
        if (isEstimate) {
            for (int i = 0; i < len_of_array; i++) {
                result += var_data[i];
            }
            result = result / len_of_array;
        }
        // Probability query
        else if (!isEstimate) {
            for (int i = 0; i < len_of_array; i++) {
                if (isMax && var_data[i] > variable_threshhold) { // Increment if value is larger than specified max
                    result += 1;
                }
                if (!isMax && var_data[i] < variable_threshhold) { // Increment if value is smaller than specified min
                    result += 1;
                }
            }
            result = result / len_of_array;
        }
        printf("Result: %lf", result);
    }
    if (variable_id == -1) {
        Statistics stats(simulations, COMP_STAT);
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
            }

    }
    // Kernels for debugging purposes
    if constexpr (VERBOSE) {
        // verify_expressions_kernel<<<1,1>>>(state);
        // test_kernel<<<1, 1>>>(state);
        // validate_edge_indices<<<1, 1>>>(state);
    }

    delete parser;
}

int main() {
    std::string filename = "../xml_files/UppaalBehaviorTest3.xml";

    int simulation_arr[] = {1, 10};

    for (int i : simulation_arr) {
        sim(filename, "", true, false, 0, 5, i);
    }


    return 1;
}
