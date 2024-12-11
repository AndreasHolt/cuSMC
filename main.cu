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

int main(int argc, char *argv[]) {
    std::string filename = "../xml_files/UppaalBehaviorTest3.xml";
    int currand_seed = 0;
    // Statistics
    int simulations = 1000;
    bool isMax = true; // Gather info on either the max value of the variable or the min

    bool succeded = HandleCommandLineArguments(argc, argv, &filename, &currand_seed, &simulations, &isMax);
    if (!succeded) {return 1;}

    const struct configuration conf = {filename, currand_seed, simulations, isMax};
    const struct model_info m_info = {64, 1};


    // Regular queries
    // string query1 = "c1.f2";
    // string query2 = "c1.f4";
    std::unordered_set<std::string> *query_set = new std::unordered_set<std::string>();
    // query_set->insert(query1);
    // query_set->insert(query2);

    // Variable queries
    int variable_id = 5; // Val : -1 implies not variable checking query

    abstract_parser *parser = new uppaal_xml_parser();


    network model = parser->parse(filename);

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
        Statistics stats(conf.simulations, VAR_STAT);

        printf("Running SMC\n");
        run_statistical_model_checking(state, 0.05, 0.01, kinds, num_vars, stats.get_comp_device_ptr(),
                                           stats.get_var_device_ptr(), variable_id, conf, m_info);
        double *var_data = stats.collect_var_data();
        for (int i = 0; i < simulations; i++) {
            printf("Variable value for simulation %d is %f\n", i, var_data[i]);
        }
    }

    // Query analysis loop
    for (auto itr = query_set->cbegin(); itr != query_set->cend(); itr++) {
        Statistics stats(conf.simulations, COMP_STAT);

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
                                           stats.get_var_device_ptr(), variable_id, conf, m_info);
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

bool HandleCommandLineArguments(int argc, char **argv, string *filename, int *seed, int * runs, bool *isMax) {
    for (int i = 1; i < argc; i++) {    // Skip first argument, which is the executable path.
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                *filename = argv[i + 1];
                i++; // Skip the next argument
            } else {
                std::cerr << "Error: -i option requires a value" << std::endl;
                return false;
            }
        } else if (arg == "-s" || arg == "--seed") {
            if (i + 1 < argc) {
                std::string str = argv[i + 1];
                i++; // Skip the next argument
                try {
                    const int num = std::stoi(str);
                    *seed = num;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: invalid argument: " << e.what() << std::endl;
                    return false;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Error: number too big: " << e.what() << std::endl;
                    return false;
                }

            } else {
                std::cerr << "Error: -s option requires a value" << std::endl;
                return false;
            }
        } else if (arg == "-r" || arg == "--runs") {
            if (i + 1 < argc) {
                std::string str = argv[i + 1];
                i++; // Skip the next argument
                try {
                    int num = std::stoi(str);
                    *runs = num;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: invalid argument: " << e.what() << std::endl;
                    return false;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Error: number too big: " << e.what() << std::endl;
                    return false;
                }

            } else {
                std::cerr << "Error: -r option requires a value" << std::endl;
                return false;
            }
        } else if (arg == "--min") {
            *isMax = false;
        } else if (arg == "--max") {
            if (*isMax == false) {
                std::cerr << "Can not use --max and --min at the same time." << std::endl;
                return false;
            }
            *isMax = true;
        } else if (arg == "-h" || arg == "--help") {
            cout << "Use -m or --model, followed by a path, for inputting a path the the model xml file." << endl;
            cout << "Use -s or --seed, followed by a number, to initialize currand with a constant seed. (0 = random seed)" << endl;
            cout << "Use -r or --runs, to specify the number of simulations." << endl;
            cout << "Use --max or --min, to specify whether we want to query for the max value a variable reaches or the lowest." << endl;
            return false;
        }
        else {
            std::cerr << "Error: unknown option " << arg << std::endl;
            return false;
        }
    }
    return true;
}
