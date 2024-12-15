//
// Created by andreas on 12/11/24.
//

#include "smc.cuh"
#include "main.cuh"
#include "simulation/write_csv.cuh"
#define TIME_BOUND 10.0


__host__ void run_statistical_model_checking(SharedModelState *model, float confidence, float precision,
                                                VariableKind *kinds, bool* flags,
                                                double* variable_flags, model_info m_info, configuration conf, statistics_Configuration stat_conf) {
    // Get max components from SharedModelState
    int MC;
    cudaMemcpy(&MC, &(model->num_components), sizeof(int), cudaMemcpyDeviceToHost);

    if constexpr (VERBOSE) {
        cout << "total_runs = " << stat_conf.simulations << endl;
    }
    // Validate parameters
    if (model == nullptr) {
        cout << "Error: NULL model pointer" << endl;
        return;
    }

    // Get device properties and validate configuration
    cudaDeviceProp deviceProp;
    cudaError_t error = cudaGetDeviceProperties(&deviceProp, 0);
    if (error != cudaSuccess) {
        cout << "Error getting device properties: " << cudaGetErrorString(error) << endl;
        return;
    }

    // Adjust threads to be multiple of warp size
    int warp_size = deviceProp.warpSize;
    //int threads_per_block = 512; // 100 components
    // int threads_per_block = ((2 + warp_size - 1) / warp_size) * warp_size; // Round up to nearest warp
    int threads_per_block = 32; // 100 components
    int runs_per_block = m_info.runs_per_block;
    int num_blocks = stat_conf.simulations;

    // Print detailed device information
    if constexpr (VERBOSE) {
        cout << "Device details:" << endl
                << "  Name: " << deviceProp.name << endl
                << "  Warp size: " << warp_size << endl
                << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << endl
                << "  Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x "
                << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << endl
                << "  Adjusted threads per block: " << threads_per_block << endl;
    }

    // Validate configuration
    if (threads_per_block > deviceProp.maxThreadsPerBlock) {
        cout << "Error: threads_per_block (" << threads_per_block
                << ") exceeds device maximum (" << deviceProp.maxThreadsPerBlock << ")" << endl;
        return;
    }

    if (num_blocks > deviceProp.maxGridSize[0]) {
        cout << "Error: num_blocks (" << num_blocks
                << ") exceeds device maximum (" << deviceProp.maxGridSize[0] << ")" << endl;
        return;
    }

    // Verify shared memory size is sufficient
    size_t shared_mem_per_block = sizeof(SharedBlockMemory);
    if (shared_mem_per_block > deviceProp.sharedMemPerBlock) {
        cout << "Error: Required shared memory (" << shared_mem_per_block
                << ") exceeds device capability (" << deviceProp.sharedMemPerBlock << ")" << endl;
        return;
    }

    // Allocate and validate device results array
    bool *device_results;
    error = cudaMalloc(&device_results, stat_conf.simulations * sizeof(bool));
    if (error != cudaSuccess) {
        cout << "CUDA malloc error: " << cudaGetErrorString(error) << endl;
        return;
    }

    if constexpr (VERBOSE) {
        cout << "Launch configuration validated:" << endl;
        cout << "  Blocks: " << num_blocks << endl;
        cout << "  Threads per block: " << threads_per_block << endl;
        cout << "  Shared memory per block: " << shared_mem_per_block << endl;
        cout << "  Time bound: " << stat_conf.timeBound << endl;
    }

    // Verify model is accessible
    SharedModelState host_model;
    error = cudaMemcpy(&host_model, model, sizeof(SharedModelState), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cout << "Error copying model: " << cudaGetErrorString(error) << endl;
        cudaFree(device_results);
        return;
    }
    if constexpr (VERBOSE) {
        cout << "Model verified accessible with " << host_model.num_components << " components" << endl;
    }
    // Add verification here with more safety checks
    if constexpr (VERBOSE) {
        cout << "\nVerifying model transfer:" << endl;
        cout << "Model contents:" << endl;
        cout << "  nodes pointer: " << host_model.nodes << endl;
        cout << "  invariants pointer: " << host_model.invariants << endl;
        cout << "  num_components: " << host_model.num_components << endl;
    }
    if (host_model.nodes == nullptr) {
        cout << "Error: Nodes array is null" << endl;
        cudaFree(device_results);
        return;
    }

    // Try to read just the pointer first
    void *nodes_ptr;
    error = cudaMemcpy(&nodes_ptr, (void *) &(model->nodes), sizeof(void *), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cout << "Error reading nodes pointer: " << cudaGetErrorString(error) << endl;
        cudaFree(device_results);
        return;
    }
    if constexpr (VERBOSE) {
        cout << "Nodes pointer verification: " << nodes_ptr << endl;
    }
    // Now try to read one node
    NodeInfo test_node;
    error = cudaMemcpy(&test_node, host_model.nodes, sizeof(NodeInfo), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cout << "Error reading node: " << cudaGetErrorString(error) << endl;
        cudaFree(device_results);
        return;
    }
    if constexpr (VERBOSE) {
        cout << "First node verification:" << endl
                << "  ID: " << test_node.id << endl
                << "  First invariant index: " << test_node.first_invariant_index << endl
                << "  Num invariants: " << test_node.num_invariants << endl;
    }
    // Only check invariants if we have a valid pointer
    if (host_model.invariants != nullptr) {
        if constexpr (VERBOSE) {
            cout << "Attempting to read invariant..." << endl;
        }
        GuardInfo test_guard;
        error = cudaMemcpy(&test_guard, host_model.invariants, sizeof(GuardInfo),
                           cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            cout << "Error reading invariant: " << cudaGetErrorString(error) << endl;
            cudaFree(device_results);
            return;
        }
        if constexpr (VERBOSE) {
            cout << "First invariant verification:" << endl
                    << "  Uses variable: " << test_guard.uses_variable << endl
                    << "  Variable ID: " << (test_guard.uses_variable ? test_guard.var_info.variable_id : -1) << endl;
        }
    } else {
        cout << "No invariants pointer available" << endl;
    }


    // Check each kernel parameter
    if constexpr (VERBOSE) {
        cout << "Kernel parameter validation:" << endl;
        cout << "  model pointer: " << model << endl;
        cout << "  device_results pointer: " << device_results << endl;
        cout << "  runs_per_block: " << runs_per_block << endl;
        cout << "  TIME_BOUND: " << TIME_BOUND << endl;
    }

    // Verify model pointer is a valid device pointer
    cudaPointerAttributes modelAttrs;
    error = cudaPointerGetAttributes(&modelAttrs, model);
    if (error != cudaSuccess) {
        cout << "Error checking model pointer: " << cudaGetErrorString(error) << endl;
        cudaFree(device_results);
        return;
    }
    if constexpr (VERBOSE) {
        cout << "Model pointer properties:" << endl;
        cout << "  type: " << (modelAttrs.type == cudaMemoryTypeDevice ? "device" : "other") << endl;
        cout << "  device: " << modelAttrs.device << endl;
    }

    // Similarly check device_results pointer
    cudaPointerAttributes resultsAttrs;
    error = cudaPointerGetAttributes(&resultsAttrs, device_results);
    if (error != cudaSuccess) {
        cout << "Error checking results pointer: " << cudaGetErrorString(error) << endl;
        cudaFree(device_results);
        return;
    }
    if constexpr (VERBOSE) {
        cout << "Results pointer properties:" << endl;
        cout << "  type: " << (resultsAttrs.type == cudaMemoryTypeDevice ? "device" : "other") << endl;
        cout << "  device: " << resultsAttrs.device << endl;
    }
    // Clear any previous error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "Previous error cleared: " << cudaGetErrorString(error) << endl;
    }

    VariableKind *d_kinds;
    error = cudaMalloc(&d_kinds, m_info.num_vars * sizeof(VariableKind)); // Assuming MAX_VARIABLES is defined
    if (error != cudaSuccess) {
        cout << "CUDA malloc error for kinds array: " << cudaGetErrorString(error) << endl;
        return;
    }

    error = cudaMemcpy(d_kinds, kinds, m_info.num_vars * sizeof(VariableKind), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cout << "Error copying kinds array: " << cudaGetErrorString(error) << endl;
        cudaFree(d_kinds);
        return;
    }
    if constexpr (VERBOSE) {
        cout << "Launching kernel..." << endl;
    }


    // RNG States
    curandState *rng_states_global;

    if constexpr (USE_GLOBAL_MEMORY_CURAND) {
        cudaMalloc(&rng_states_global, MC * sizeof(curandState));
    }


    cudaGetDeviceProperties(&deviceProp, 0); // Assuming device 0, change if necessary



    // What share memory we need:
    /*
    __shared__ double delays[MAX_COMPONENTS]; // Only need MAX_COMPONENTS slots, not full warp size
    __shared__ int component_indices[MAX_COMPONENTS];
    __shared__ SharedBlockMemory shared_mem;
    __shared__ ComponentState components[MAX_COMPONENTS];
    curandState *rng_states;

    // Store curandStates in either global memory or shared memory. Requires ~90kb of shared memory w/ curandStates
    if constexpr (USE_GLOBAL_MEMORY_CURAND) {
        // extern __shared__ curandState *rng_states_global;
        rng_states = rng_states_global;
    } else {
        __shared__ curandState rng_states_shared[MAX_COMPONENTS];
        rng_states = rng_states_shared;
    }



    */

    const chrono::steady_clock::time_point global_start = chrono::steady_clock::now();
    // Dynamic Shared memory: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
    // int MC = m_info.MAX_COMPONENTS;
    if constexpr (USE_GLOBAL_MEMORY_CURAND) {
        simulation_kernel<<<num_blocks, threads_per_block, MC*sizeof(double) + MC*sizeof(int) + sizeof(SharedBlockMemory) + MC*sizeof(ComponentState)>>>(
        model, device_results, runs_per_block, stat_conf.timeBound, d_kinds, m_info.num_vars, flags, variable_flags, stat_conf.variable_id, stat_conf.isMax, rng_states_global, conf.curand_seed, MC);
    } else {
        simulation_kernel<<<num_blocks, threads_per_block, MC*sizeof(double) + MC*sizeof(int) + sizeof(SharedBlockMemory) + MC*sizeof(ComponentState) + MC*sizeof(curandState)>>>(
        model, device_results, runs_per_block, stat_conf.timeBound, d_kinds, m_info.num_vars, flags, variable_flags, stat_conf.variable_id, stat_conf.isMax, rng_states_global, conf.curand_seed, MC);
    }

    cudaDeviceSynchronize();




    auto duration_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - global_start);
    long long nanoseconds = duration_nano.count();
    double ms = duration_nano.count() / 1000000.0;
    cout << "Time taken: " << ms << " ms on file " << conf.filename << endl;
    cout << "Time taken: " << nanoseconds << " on file " << conf.filename << endl;
    std::cout << "Time taken: " << duration_nano.count() << " ns ("
          << (duration_nano.count() / 1000000.0) << " ms)" << std::endl;

    writeTimingToCSV(conf.filename, MC, stat_conf.simulations, stat_conf.timeBound, ms);



    // Check for launch error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "Launch error: " << cudaGetErrorString(error) << endl;
        cudaFree(device_results);
        return;
    }

    // Check for execution error
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        cout << "Execution error: " << cudaGetErrorString(error) << endl;
        cudaFree(device_results);
        return;
    }
    if constexpr (VERBOSE) {
        cout << "Kernel completed successfully" << endl;
    }

    // Cleanup
    cudaFree(d_kinds);
    cudaFree(device_results);
}

void smc(configuration conf, statistics_Configuration stat_conf) {
    // Read and parse XML file
    abstract_parser *parser = new uppaal_xml_parser();
    network model = parser->parse(conf.filename);
    network_props properties = {};
    populate_properties(properties, parser);

    std::unordered_set<std::string> *query_set = new std::unordered_set<std::string>();
    query_set->insert(stat_conf.loc_query);
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

    if (PRINT_VARIABLES) {
        auto node_name = parser->get_nodes_with_name();
        auto variable_name = parser->get_variables_names_to_ids_map();
        for (auto iter = node_name->begin(); iter != node_name->end(); ++iter) {
            auto relation = *iter;
            cout << "Node id: " << relation.first << " name: " << relation.second << endl;
        }
        for (auto iter = variable_name.begin(); iter != variable_name.end(); ++iter) {
            auto relation = *iter;
            cout << "Variable id: " << relation.first << " name: " << relation.second << endl;
        }


    }

    VariableKind *kinds = var_tracker.createKindArray(registry);
    uint32_t num_vars = registry.size();

    // TODO: Calculate these values... (MAX VALUE STACK = FANOUT?
    const struct model_info m_info = { 64, 1, num_vars};


    cout << "=================\n\n";
    cout << "Running SMC with the following settings:" << std::endl;
    cout << "- Number of simulations: " << stat_conf.simulations << std::endl;
    cout << "- Model: " << conf.filename << std::endl;
    cout << "- Random seed: " << conf.curand_seed << std::endl;
    cout << "- Time bound: " << stat_conf.timeBound << std::endl;

    if (!stat_conf.loc_query.empty()) {
        cout << "- Logging type: \"comp.node\" query" << std::endl;
    } else if (stat_conf.variable_id != -1) {
        string min_or_max = (stat_conf.isMax) ? "max" : "min";
        cout << "- Logging type: variable query. Finding " << min_or_max
             << " of variable with ID " << stat_conf.variable_id << std::endl;
    }

    cout << "=================\n\n";

    double result = 0;
    // Handling variable queries
    if (stat_conf.variable_id != -1) {
        std::unordered_map<int, node *> node_map = optimizer.get_node_map();
        SharedModelState *state = init_shared_model_state(
            &model, // cpu_network
            *optimizer.get_node_subsystems_map(),
            *properties.node_edge_map,
            node_map,
            var_tracker.get_variable_registry(),
            parser,
            m_info.num_vars);
        Statistics stats(stat_conf.simulations, VAR_STAT);


        run_statistical_model_checking(state, 0.05, 0.01, kinds, stats.get_comp_device_ptr(),
                                           stats.get_var_device_ptr(), m_info, conf, stat_conf);

        int len_of_array = stat_conf.simulations;
        double *var_data = stats.collect_var_data();
        // Estimate query
        if (stat_conf.isEstimate) {
            for (int i = 0; i < len_of_array; i++) {
                cout << "Adding " << var_data[i] << " to sum " << result << endl;
                result += var_data[i];
            }
            result = result / len_of_array;
        }
        // Probability query
        else if (!stat_conf.isEstimate) {
            for (int i = 0; i < len_of_array; i++) {
                if (stat_conf.isMax && var_data[i] > stat_conf.variable_threshhold) { // Increment if value is larger than specified max
                    result += 1;
                }
                if (!stat_conf.isMax && var_data[i] < stat_conf.variable_threshhold) { // Increment if value is smaller than specified min
                    result += 1;
                }
            }
            result = result / len_of_array;
        }
        std::cout << "Result: " << result << endl;
    }

    if (stat_conf.variable_id == -1) {
        Statistics stats(stat_conf.simulations, COMP_STAT);
        if constexpr (VERBOSE) {
            cout << "Recorded query is: " + stat_conf.loc_query << endl;
        }

        // String split
        std::vector<char> component;
        std::vector<char> goal_node;
        bool period_reached = false;
        for (int i = 0; i < stat_conf.loc_query.length(); i++) {
            // Guard
            if (stat_conf.loc_query[i] == '.') {
                period_reached = true;
                continue;
            }
            if (!period_reached) {
                component.push_back(stat_conf.loc_query[i]);
            }
            if (period_reached) {
                goal_node.push_back(stat_conf.loc_query[i]);
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
        run_statistical_model_checking(state, 0.05, 0.01, kinds, stats.get_comp_device_ptr(),
                                           stats.get_var_device_ptr(), m_info, conf, stat_conf);
        try {
            auto results = stats.collect_results();
            stats.print_results(stat_conf.loc_query, results);
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
