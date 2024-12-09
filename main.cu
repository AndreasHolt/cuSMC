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
    // string query1 = "c1.f2";
    string query2 = "c1.f4";
    std::unordered_set<std::string>* query_set = new std::unordered_set<std::string>();
    // query_set->insert(query1);
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


    std::unordered_map<std::string, int> template_name_int_map;
    for (auto itr = properties.template_names->cbegin(); itr != properties.template_names->cend(); itr++) {
        template_name_int_map.insert({itr->second, itr->first});
    }

    std::unordered_map<std::string, int> node_name_int_map;
    for (auto itr = properties.node_names->cbegin(); itr != properties.node_names->cend(); itr++) {
        node_name_int_map.insert({itr->second, itr->first});
    }


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

    /* SharedModelState* state = init_shared_model_state(
    &model,  // cpu_network
    *optimizer.get_node_subsystems_map(),
    *properties.node_edge_map,
    optimizer.get_node_map(),
    var_tracker.get_variable_registry(),
    parser, num_vars
    );*/

    // Simulation count
    // size_t simulations = config.total_simulations();
    int simulations = 10000;


    // Query analysis loop (Sidenote: Fuck unordered sets)
    for (auto itr = query_set->cbegin(); itr != query_set->cend(); itr++){
        // Query string
        string query = *(*query_set).find(*itr);

        if constexpr (VERBOSE){
            cout << "Recorded query is: " + query << endl;
        }
        // Variable queries:
        int variable_id = -1; //Val : -1 implies not variable checking query
        // IF Query : refs : variable {
        //      x <- Find variable id from name
        //      variable_id = x
        //  }

        // String split
        std::vector<char> component;
        std::vector<char> goal_node;
        bool period_reached = false;
        for (int i = 0; i < query.length(); i++){
            // Guard
            if (query[i] == '.'){period_reached = true; continue;}
            if (!period_reached) {
                component.push_back(query[i]);
            }
            if (period_reached) {
                goal_node.push_back(query[i]);
            }
        }
        std::string component_name(component.begin(), component.end());
        //template_name_int_map.find(component_name);

        std::string node_name(goal_node.begin(), goal_node.end());
        auto temp = node_name_int_map.find(node_name);
        std::unordered_map<int, node*> node_map = optimizer.get_node_map();

        if (temp != node_name_int_map.cend()) {
            int goal_node_idx = (*temp).second;

            (*node_map.find(goal_node_idx)).second->type = node::goal;
        }

        SharedModelState* state = init_shared_model_state(
            &model,  // cpu_network
            *optimizer.get_node_subsystems_map(),
            *properties.node_edge_map,
            node_map,
            var_tracker.get_variable_registry(),
            parser,
            num_vars);


        cudaError_t error;

        bool* goal_flags_host_ptr = (bool*)malloc(simulations*sizeof(bool));
        if (!goal_flags_host_ptr) {
            cout << "Error: Host memory allocation failed" << endl;
            return 0;
        }

        memset(goal_flags_host_ptr, 0, simulations * sizeof(bool));



        bool* goal_flags_device_ptr;
        error = cudaMalloc(&goal_flags_device_ptr, simulations*sizeof(bool));
        if (error != cudaSuccess) {
            cout << "Error: Device memory allocation failed: " << cudaGetErrorString(error) << endl;
            free(goal_flags_host_ptr);
            return 0;
        }

        error = cudaMemcpy(goal_flags_device_ptr, goal_flags_host_ptr, simulations*sizeof(bool), cudaMemcpyHostToDevice);

        // Run the SMC simulations
        sim.run_statistical_model_checking(state, 0.05, 0.01, kinds, num_vars, goal_flags_device_ptr, variable_id, simulations);

        error = cudaMemcpy(goal_flags_host_ptr, goal_flags_device_ptr, simulations*sizeof(bool), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            cout << "Error: Memory copy from device failed: " << cudaGetErrorString(error) << endl;
            free(goal_flags_host_ptr);
            cudaFree(goal_flags_device_ptr);
            return 0;
        }

        int hits = 0;
        for (int i = 0; i < simulations; i++) {
            printf("Flag in host is %d\n", goal_flags_host_ptr[i]);
            if (goal_flags_host_ptr[i]) {hits += 1;}
        }

        free(goal_flags_host_ptr);
        cudaFree(goal_flags_device_ptr);

        float res = static_cast<float>(hits) / static_cast<float>(simulations);

        cout << "Number of simulations: " << simulations << endl;
        cout << "Number of hits: " << hits << endl;
        string output = "The answer to " + query + " is " + std::to_string(res);

        cout << output << endl;
    }


    // Kernels for debugging purposes
    if constexpr (VERBOSE) {
        // verify_expressions_kernel<<<1,1>>>(state);
        // cudaDeviceSynchronize();
        // verify_invariants_kernel<<<1, 1>>>(state);
        // test_kernel<<<1, 1>>>(state);
        // validate_edge_indices<<<1, 1>>>(state);
    }

    delete parser;

    return 0;
}
