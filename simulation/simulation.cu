// Created by andwh on 24/10/2024.

#include "simulation.cuh"
#include "expressions.cuh"

#define CHECK_ERROR(loc) check_cuda_error(loc)

// Calculate sum of edges from enabled set
__device__ int get_sum_edge_weight(const ComponentState *my_state,
                                   const SharedModelState *model,
                                   SharedBlockMemory *shared) {
    double sum = 0;
    const NodeInfo *node = my_state->current_node;
    for (int i = 0; i < node->num_edges; i++) {
        if (my_state->enabled_edges[i] + node->first_edge_index == node->first_edge_index + i) {
            sum += evaluate_expression(model->edges[node->first_edge_index + i].weight, shared);
        }
    }
    return static_cast<int>(sum);
}


__device__ void take_transition(ComponentState *my_state,
                                SharedBlockMemory *shared,
                                SharedModelState *model,
                                BlockSimulationState *block_state, int query_variable_id) {
    if (my_state->num_enabled_edges == 0) {
        if constexpr (MINIMAL_PRINTS) {
            printf("Thread %d: No enabled edges to take\n", threadIdx.x);
        }
        return;
    }

    // Select random edge from enabled ones
    int selected_idx;
    if (my_state->num_enabled_edges == 1) {
        selected_idx = my_state->enabled_edges[0];
        if constexpr (VERBOSE) {
            printf("Thread %d: Only one enabled edge (%d), selecting it\n",
                   threadIdx.x, selected_idx);
        }
    } else {
        // Random selection between enabled edges
        const int weights = get_sum_edge_weight(my_state, model, shared);
        float random = curand_uniform(block_state->random);
        const int rand = static_cast<int>(static_cast<float>(weights) * random);
        int temp = weights;
        const NodeInfo *node = my_state->current_node;

        for (int i = my_state->num_enabled_edges - 1; i >= 0; i--) {
            int weight_of_edge = static_cast<int>(
                evaluate_expression(model->edges[node->first_edge_index + i].weight, shared));
            if (temp - weight_of_edge <= rand) {
                selected_idx = my_state->enabled_edges[i];
                break;
            }
            temp -= weight_of_edge;
        }

        if constexpr (VERBOSE) {
            printf("Thread %d: Randomly selected edge %d from %d enabled edges (rand=%d)\n",
                   threadIdx.x, selected_idx, my_state->num_enabled_edges, rand);
        }
    }

    // Get the selected edge
    const EdgeInfo &edge = model->edges[my_state->current_node->first_edge_index + selected_idx];
    if constexpr (VERBOSE) {
        printf("Thread %d: Taking transition from node %d to node %d\n",
               threadIdx.x, my_state->current_node->id, edge.dest_node_id);
    }

    // If this edge has a positive channel (broadcast sender)
    if (edge.channel > 0) {
        int channel_abs = abs(edge.channel);

        // This component needs to signal the broadcast, as its channel is '!-labelled'
        shared->channel_active[channel_abs] = true;
        shared->channel_sender[channel_abs] = my_state->component_id;
    }


    // Apply updates if any
    for (int i = 0; i < edge.num_updates; i++) {
        const UpdateInfo &update = model->updates[edge.updates_start_index + i];
        int var_id = update.variable_id;

        // Evaluate update expression
        double new_value = evaluate_expression(update.expression, shared);
        if constexpr (VERBOSE) {
            printf("Thread %d: Update %d - Setting var_%d (%s) from %f to %f\n",
                   threadIdx.x, i, var_id,
                   update.kind == VariableKind::CLOCK ? "clock" : "int",
                   shared->variables[var_id].value,
                   new_value);
        }

        if (var_id == query_variable_id) {
            if (new_value > shared->query_variable_max) {
                if constexpr (VERBOSE) {
                    printf("Changing max to %f\n", new_value);
                }
                shared->query_variable_max = new_value;
            }
            if (new_value < shared->query_variable_min) {
                if constexpr (VERBOSE) {
                    printf("Changing min to %f\n", new_value);
                }
                shared->query_variable_min = new_value;
            }
        }

        shared->variables[var_id].value = new_value;
        shared->variables[var_id].last_writer = my_state->component_id;
    }

    // Find destination node info
    // First find node in the same level that matches destination ID
    if constexpr (VERBOSE) {
        printf("Thread %d: Searching for destination node %d (current level=%d)\n",
               threadIdx.x, edge.dest_node_id, my_state->current_node->level);
    }

    const NodeInfo *dest_node = nullptr;

    // If there are multiple runs in 1 block, then we can still find relevant index by using the modulo component
    // Example: 4 runs, thread 1, 26, 51, 76 all access index 1. Aka. modulo num_components
    for (int level = 0; level < model->max_nodes_per_component; level++) {
        // We know relevant nodes are stored at each level offset by the index of the component idx
        // The component idx is equivalent to threadIdx
        const NodeInfo &node = model->nodes[level * model->num_components + threadIdx.x];
        if (node.id == edge.dest_node_id) {
            dest_node = &node;

            if (dest_node->type == 1) {
                shared->has_hit_goal = true;
                if constexpr (QUERYSTATS) {
                    printf("Block %d thread %d has reached the goal state at node %d\n", blockIdx.x, threadIdx.x,
                           edge.dest_node_id);
                }
            }

            if constexpr (VERBOSE) {
                printf("Thread %d: Found destination node at level %d, index %d\n",
                       threadIdx.x, level, (threadIdx.x));
            }
            break;
        }
    }

    if (dest_node == nullptr) {
        printf("Thread %d: ERROR - Could not find destination node %d!\n",
               threadIdx.x, edge.dest_node_id);
        return;
    }

    // Update current node
    my_state->current_node = dest_node;
    if constexpr (VERBOSE) {
        printf("Thread %d: Moved to new node %d\n", threadIdx.x, dest_node->id);
    }
}


__device__ bool check_edge_enabled(const EdgeInfo &edge,
                                   SharedBlockMemory *shared,
                                   SharedModelState *model,
                                   BlockSimulationState *block_state, bool is_broadcast_sync) {
    if constexpr (VERBOSE) {
        printf("\nThread %d: Checking edge %d->%d with %d guards\n",
               threadIdx.x, edge.source_node_id, edge.dest_node_id, edge.num_guards);
    }

    // Only reject negative channels if not part of broadcast sync
    if (edge.channel < 0 && !is_broadcast_sync) {
        if constexpr (VERBOSE) {
            printf("Thread %d: Is a !-labelled channel, disabled until synchronisation\n", threadIdx.x);
        }
        return false;
    }

    // Check all guards on the edge
    for (int i = 0; i < edge.num_guards; i++) {
        const GuardInfo &guard = model->guards[edge.guards_start_index + i];

        if (guard.uses_variable) {
            int var_id = guard.var_info.variable_id;
            double var_value = shared->variables[var_id].value;
            double bound = evaluate_expression(guard.expression, shared);
            if constexpr (VERBOSE) {
                printf("  Guard %d: var_%d (%s) = %f %s %f\n",
                       i, var_id,
                       guard.var_info.type == VariableKind::CLOCK ? "clock" : "int",
                       var_value,
                       guard.operand == constraint::less_equal_c
                           ? "<="
                           : guard.operand == constraint::less_c
                                 ? "<"
                                 : guard.operand == constraint::greater_equal_c
                                       ? ">="
                                       : guard.operand == constraint::greater_c
                                             ? ">"
                                             : "?",
                       bound);
            }

            bool satisfied = false;
            switch (guard.operand) {
                case constraint::less_c:
                    satisfied = var_value < bound;
                    break;
                case constraint::less_equal_c:
                    satisfied = var_value <= bound;
                    break;
                case constraint::greater_c:
                    satisfied = var_value > bound;
                    break;
                case constraint::greater_equal_c:
                    satisfied = var_value >= bound;
                    break;
                default:
                    printf("  Warning: Unknown operator %d\n", guard.operand);
                    return false;
            }

            if (!satisfied) {
                if constexpr (VERBOSE) {
                    printf("  Guard not satisfied - edge disabled\n");
                }
                return false;
            }
        }
    }
    if constexpr (VERBOSE) {
        printf("  All guards satisfied - edge enabled!\n");
    }
    return true;
}

__device__ void check_enabled_edges(ComponentState *my_state,
                                    SharedBlockMemory *shared,
                                    SharedModelState *model,
                                    BlockSimulationState *block_state,
                                    bool is_race_winner) {
    if (!is_race_winner) {
        if constexpr (VERBOSE) {
            printf("Thread %d: Skipping edge check (didn't win race)\n", threadIdx.x);
        }
        return;
    }
    if constexpr (VERBOSE) {
        printf("\nThread %d: Checking enabled edges for node %d\n",
               threadIdx.x, my_state->current_node->id);
    }

    const NodeInfo &node = *my_state->current_node;
    my_state->num_enabled_edges = 0; // Reset counter

    // Check each outgoing edge
    for (int i = 0; i < node.num_edges; i++) {
        const EdgeInfo &edge = model->edges[node.first_edge_index + i];
        if (check_edge_enabled(edge, shared, model, block_state, false)) {
            // Store enabled edge for later selection
            my_state->enabled_edges[my_state->num_enabled_edges++] = i;
            if constexpr (VERBOSE) {
                printf("Thread %d: Edge %d is enabled (total enabled: %d)\n",
                       threadIdx.x, i, my_state->num_enabled_edges);
            }
        }
    }
    if constexpr (VERBOSE) {
        printf("Thread %d: Found %d enabled edges\n",
               threadIdx.x, my_state->num_enabled_edges);
    }
}


__device__ void check_cuda_error(const char *location) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error at %s: %s\n", location, cudaGetErrorString(error));
    }
}


__device__ void compute_possible_delay(
    ComponentState *my_state,
    SharedBlockMemory *shared,
    SharedModelState *model,
    BlockSimulationState *block_state, int num_vars, int query_variable_id) {
    // First check for active broadcasts
    for (int ch = 0; ch < MAX_CHANNELS; ch++) {
        if (shared->channel_active[ch] &&
            shared->channel_sender[ch] != my_state->component_id) {
            // This is the correct check

            // Collect all enabled receiving edges
            my_state->num_enabled_edges = 0;
            const NodeInfo *current_node = my_state->current_node;

            for (int e = 0; e < current_node->num_edges; e++) {
                const EdgeInfo &edge = model->edges[current_node->first_edge_index + e];
                if (edge.channel == -ch &&
                    check_edge_enabled(edge, shared, model, block_state, true)) {
                    // Edges is enabled, therefore we add it to the enabled edges list, to be used inside take_transition
                    my_state->enabled_edges[my_state->num_enabled_edges++] = e;
                }
            }

            // If we found any enabled receiving edges, we just randomly select one inside take_transition
            if (my_state->num_enabled_edges > 0) {
                if constexpr (VERBOSE) {
                    printf("Found %d enabled receiving edges for channel %d.\n",
                           my_state->num_enabled_edges, ch);
                }
                take_transition(my_state, shared, model, block_state, query_variable_id);
            }
        }
    }

    __syncthreads();

    // Only a single thread needs to reset synchronisation flags in shared memory
    if (threadIdx.x == 0) {
        for (int ch = 0; ch < MAX_CHANNELS; ch++) {
            shared->channel_active[ch] = false;
            shared->channel_sender[ch] = -1;
        }
    }

    __syncthreads();

    const NodeInfo &node = *my_state->current_node;
    if constexpr (VERBOSE) {
        printf("Thread %d: Processing node %d with %d invariants\n",
               threadIdx.x, node.id, node.num_invariants);
        printf("Lambda on node %d is polish notation: %s\n", node.id,
               node.lambda->operand == expr::pn_compiled_ee ? "true" : "false");
    }

    double min_delay = 0.0;
    double max_delay = DBL_MAX;
    bool is_bounded = false;

    __syncthreads();

    if constexpr (VERBOSE) {
        printf("Node idx %d has type: %d \n", node.id, node.type);
    }

    // Node types with 3 (Urgent) or 4 (Commited) need to return 0 as their delay (they are immediate)
    if (node.type > 2) {
        if constexpr (VERBOSE) {
            printf("Node idx %d has type: %d, therefore it is urgent or commited and selecting delay 0 \n", node.id,
                   node.type);
        }
        my_state->next_delay = 0;
        my_state->has_delay = true;
        return;
    }

    // Debug current variable values
    if constexpr (VERBOSE) {
        if (threadIdx.x == 0) {
            printf("Thread %d: Current variable values:\n", threadIdx.x);
            for (int i = 0; i < num_vars; i++) {
                printf("  var[%d] = %f (rate=%d)\n", i,
                       shared->variables[i].value,
                       shared->variables[i].rate);
            }
        }
    }

    // Process invariants
    for (int i = 0; i < node.num_invariants; i++) {
        const GuardInfo &inv = model->invariants[node.first_invariant_index + i];

        if (inv.uses_variable) {
            int var_id = inv.var_info.variable_id;
            if (var_id >= num_vars) {
                printf("Thread %d: Invalid variable ID %d\n", threadIdx.x, var_id);
                continue;
            }

            auto &var = shared->variables[var_id];
            double current_val = var.value;

            // Set rate to 1 for clocks
            if (inv.var_info.type == VariableKind::CLOCK) {
                // TODO: fetch the rate from the model. Used for exponential distribution etc.
                var.rate = 1;
            }

            // Evaluate bound expression
            double bound = evaluate_expression(inv.expression, shared);
            if constexpr (VERBOSE) {
                printf("Thread %d: Clock %d invariant: current=%f, bound=%f, rate=%d\n",
                       threadIdx.x, var_id, current_val, bound,
                       var.rate); // TODO: remove rate from var. Rates are dependent on the location
            }
            // Only handle upper bounds
            if (inv.operand == constraint::less_c ||
                inv.operand == constraint::less_equal_c) {
                if (var.rate > 0) {
                    // Only if clock increases
                    double time_to_bound = (bound - current_val) / var.rate;

                    // Add small epsilon for strict inequality
                    if (inv.operand == constraint::less_c) {
                        time_to_bound -= 1e-6;
                    }

                    if constexpr (VERBOSE) {
                        printf("Thread %d: Computed time_to_bound=%f\n",
                               threadIdx.x, time_to_bound);
                    }

                    if (time_to_bound >= 0) {
                        max_delay = min(max_delay, time_to_bound);
                        // TODO: remove time_to_bound, as it is not part of the semantics
                        is_bounded = true;
                        if constexpr (VERBOSE) {
                            printf("Thread %d: Updated max_delay to %f\n",
                                   threadIdx.x, max_delay);
                        }
                    }
                }
            }
        }
    }

    // Sample delay if bounded
    if (is_bounded) {
        double rand = curand_uniform(block_state->random);
        my_state->next_delay = min_delay + (max_delay - min_delay) * rand;
        my_state->has_delay = true;
        if constexpr (VERBOSE) {
            printf("Thread %d: Sampled delay %f in [%f, %f] (rand=%f)\n",
                   threadIdx.x, my_state->next_delay, min_delay, max_delay, rand);
        }
    } else {
        double rate = 1.0; // Default rate if no rate is specified on the node
        if (node.lambda != nullptr) {
            rate = evaluate_expression(node.lambda, shared);
        }

        // Sample from the exponential distribution
        double rand = curand_uniform_double(block_state->random);
        my_state->next_delay = -__log2f(rand) / rate;
        // Fastest log, but not as accurate. We consider it fine because we are doing statistical sampling

        my_state->has_delay = true;
        if constexpr (VERBOSE) {
            printf("Thread %d: No delay bounds, sampled %f using exponential distribution with rate %f\n", threadIdx.x,
                   my_state->next_delay, rate);
        }
        my_state->has_delay = true;
    }
}

// Finds the minimum delay and the winning component takes a transition
__device__ double find_minimum_delay(
    ComponentState *my_state,
    SharedBlockMemory *shared,
    SharedModelState *model,
    BlockSimulationState *block_state,
    int num_components, int query_variable_id,
    double *delays, int *component_indices) {
    //__shared__ double delays[MAX_COMPONENTS]; // Only need MAX_COMPONENTS slots, not full warp size
    //__shared__ int component_indices[MAX_COMPONENTS];

    // Initialize only for active threads (components)
    if (threadIdx.x < num_components) {
        delays[threadIdx.x] = my_state->has_delay ? my_state->next_delay : DBL_MAX;
        component_indices[threadIdx.x] = my_state->component_id;
        if constexpr (VERBOSE) {
            printf("Thread %d (component %d): Initial delay %f\n",
                   threadIdx.x, my_state->component_id, delays[threadIdx.x]);
        }
    }
    __syncthreads();

    // Debug print initial state
    if constexpr (VERBOSE) {
        if (threadIdx.x == 0) {
            printf("Initial delays: ");
            for (int i = 0; i < num_components; i++) {
                // Only print actual components
                printf("[%d]=%f ", i, delays[i]);
            }
            printf("\n");
        }
        __syncthreads();
    }

    // Find minimum - only compare actual components
    for (int stride = (num_components + 1) / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x < num_components) {
            int compare_idx = threadIdx.x + stride;
            if (compare_idx < num_components) {
                // Only compare if within valid components
                if constexpr (VERBOSE) {
                    printf("Thread %d comparing with position %d: %f vs %f\n",
                           threadIdx.x, compare_idx, delays[threadIdx.x],
                           delays[compare_idx]);
                }

                if (delays[compare_idx] < delays[threadIdx.x]) {
                    delays[threadIdx.x] = delays[compare_idx];
                    component_indices[threadIdx.x] = component_indices[compare_idx];
                    if constexpr (VERBOSE) {
                        printf("Thread %d: Updated minimum to %f from component %d\n",
                               threadIdx.x, delays[threadIdx.x],
                               component_indices[threadIdx.x]);
                    }
                }
            }
        }
        __syncthreads();
    }

    double min_delay = delays[0];
    if (threadIdx.x == 0) {
        if (min_delay < DBL_MAX) {
            if constexpr (VERBOSE) {
                printf("\nFinal result:\n");
                printf("  Minimum delay: %f\n", min_delay);
                printf("  Winning component: %d\n", component_indices[0]);
                printf("\nUpdating clocks:\n");
            }

            for (int i = 0; i < MAX_VARIABLES; i++) {
                if (shared->variables[i].kind == VariableKind::CLOCK) {
                    double old_value = shared->variables[i].value;
                    shared->variables[i].rate = 1;
                    // TODO: Don't do this. Also why do variables have a rate? Locations have rates.
                    shared->variables[i].value += min_delay;
                    if constexpr (VERBOSE) {
                        printf("  Clock %d: %f -> %f (advanced by %f)\n",
                               i, old_value, shared->variables[i].value, min_delay);
                    }
                }
            }
        }
    }
    __syncthreads();

    // Remember who won and check edges only for winner
    bool is_race_winner = false;
    if (threadIdx.x < num_components) {
        bool is_race_winner = (min_delay < DBL_MAX &&
                               component_indices[0] == my_state->component_id);

        if (is_race_winner) {
            check_enabled_edges(my_state, shared, model, block_state, is_race_winner);
            take_transition(my_state, shared, model, block_state, query_variable_id);
        }
    }

    __syncthreads();

    return min_delay;
}

__global__ void simulation_kernel(SharedModelState *model, bool *results,
                                  int runs_per_block, float time_bound, VariableKind *kinds, uint num_vars, bool *flags,
                                  double *variable_flags, int variable_id, bool isMax,
                                  curandState *rng_states_global, int curand_seed, int max_components) {
    // Prepare Shared memory
    //extern __shared__ int s[];
    //int *integerData = s;                        // nI ints
    //float *floatData = (float*)&integerData[nI]; // nF floats
    //char *charData = (char*)&floatData[nF];      // nC chars
    extern __shared__ int s[];

    SharedBlockMemory *shared_mem = (SharedBlockMemory *) &s;
    // Using (char*) because it is 1 byte large.
    ComponentState *components = (ComponentState *) ((char *) shared_mem + sizeof(SharedBlockMemory));

    double *delays = (double *) &components[max_components]; // Only need MAX_COMPONENTS slots, not full warp size
    int *component_indices = (int *) &delays[max_components];

    curandState *rng_states;

    // Store curandStates in either global memory or shared memory. Requires ~90kb of shared memory w/ curandStates
    if constexpr (USE_GLOBAL_MEMORY_CURAND) {
        // extern __shared__ curandState *rng_states_global;
        rng_states = rng_states_global;
    } else {
        curandState *rng_states_shared = (curandState *) component_indices[max_components];
        rng_states = rng_states_shared;
    }

    CHECK_ERROR("Kernel start");
    if constexpr (VERBOSE) {
        if (threadIdx.x == 0) {
            printf("Starting kernel: block=%d, thread=%d\n",
                   blockIdx.x, threadIdx.x);
            printf("Number of variables: %d\n", num_vars);
        }

        // Verify model pointer
        if (model == nullptr) {
            printf("Thread %d: NULL model pointer!\n", threadIdx.x);
            return;
        }
    }


    if constexpr (VERBOSE) {
        if (threadIdx.x < model->num_components) {
            // Only debug print for actual components
            printf("Thread %d: Model details:\n"
                   "  Num components: %d\n"
                   "  First node invariant index: %d\n"
                   "  Num invariants in first node: %d\n",
                   threadIdx.x,
                   model->num_components,
                   model->nodes[threadIdx.x].first_invariant_index,
                   model->nodes[threadIdx.x].num_invariants);
        }
    }

    __syncthreads();

    CHECK_ERROR("after shared memory declaration");

    // Debug model access
    if (threadIdx.x == 0) {
        if constexpr (VERBOSE) {
            printf("Thread %d: Attempting to access model, num_components=%d\n",
                   threadIdx.x, model->num_components);
        }
    }

    CHECK_ERROR("after model access");


    // Setup block state
    BlockSimulationState block_state;

    block_state.my_component = &components[threadIdx.x];
    if constexpr (VERBOSE) {
        if (threadIdx.x == 0) {
            printf("Thread %d: Block state setup complete\n", threadIdx.x);
        }
    }

    CHECK_ERROR("after block state setup");

    // Initialize RNG
    int sim_id = blockIdx.x * runs_per_block;
    int comp_id = threadIdx.x;
    curand_init(curand_seed + sim_id * blockDim.x + comp_id, 0, 0,
                &rng_states[threadIdx.x]);
    block_state.random = &rng_states[threadIdx.x];

    // printf("Thread %d: RNG initialized\n", threadIdx.x);
    CHECK_ERROR("after RNG init");

    // Initialize shared state
    if (threadIdx.x == 0) {
        if constexpr (VERBOSE) {
            printf("Block %d: Initializing shared memory\n", blockIdx.x);
        }
        SharedBlockMemory::init(shared_mem, sim_id);

        for (int i = 0; i < num_vars; i++) {
            if constexpr (VERBOSE) {
                printf("Setting variable %d to kind %d\n", i, kinds[i]);
            }
            shared_mem->variables[i].value = model->initial_var_values[i];
            if (kinds[i] == VariableKind::CLOCK) {
                shared_mem->variables[i].kind = VariableKind::CLOCK;
            } else if (kinds[i] == VariableKind::INT) {
                shared_mem->query_variable_min = model->initial_var_values[i];
                shared_mem->query_variable_max = model->initial_var_values[i];
                // Not sure whether we need this case yet
            }
        }
    }
    __syncthreads();
    CHECK_ERROR("after shared memory init");

    // Initialize component state
    // TODO: Move this to the top of the function?
    if (threadIdx.x >= model->num_components) {
        if constexpr (MINIMAL_PRINTS) {
            printf("Thread %d: Exiting - thread ID exceeds number of components\n",
                   threadIdx.x);
        }
        return;
    }

    ComponentState *my_state = block_state.my_component;
    my_state->component_id = comp_id;
    my_state->current_node = &model->nodes[comp_id];
    my_state->has_delay = false;
    if constexpr (VERBOSE) {
        printf("Thread %d: Component initialized, node_id=%d\n",
               threadIdx.x, my_state->current_node->id);
    }
    CHECK_ERROR("after component init");

    // Main simulation loop
    while (shared_mem->global_time < time_bound) {
        __syncthreads();
        // Synchronize before continuing to make sure all threads have the latest value of shared_mem.has_hit_goal etc.

        if (shared_mem->has_hit_goal && flags != nullptr) {
            // All threads should check whether the goal has been reached
            if (threadIdx.x == 0) {
                if constexpr (QUERYSTATS) {
                    printf("Flag was true for block %d\n", blockIdx.x);
                }
                flags[blockIdx.x] = true;
                // ... but only a single thread should write to the flag to avoid race conditions
            }
            break;
        }


        if constexpr (VERBOSE) {
            printf("Flag is %d\n", shared_mem->has_hit_goal);
            printf("Thread %d: Time=%f\n", threadIdx.x, shared_mem->global_time);
        }
        compute_possible_delay(my_state, shared_mem, model, &block_state, num_vars, variable_id);

        CHECK_ERROR("after compute delay");
        __syncthreads();

        double min_delay = find_minimum_delay(
            block_state.my_component, // ComponentState*
            shared_mem, // SharedBlockMemory*
            model, // SharedModelState*
            &block_state, // BlockSimulationState*
            model->num_components, // int num_components
            variable_id,
            delays, //Delays in shared memory
            component_indices //Component indices in shared memory.
        );
        CHECK_ERROR("after find minimum");
        if constexpr (VERBOSE) {
            printf("Thread %d: Minimum delay = %f\n", threadIdx.x, min_delay);
        }
        if (threadIdx.x == 0) {
            shared_mem->global_time += min_delay;
            if constexpr (VERBOSE) {
                printf("Block %d: Advanced time to %f\n",
                       blockIdx.x, shared_mem->global_time);
            }
        }

        __syncthreads(); // Sync to make sure all threads see the break condition
    }

    if (variable_flags != nullptr) {
        if (isMax) {
            variable_flags[blockIdx.x] = shared_mem->query_variable_max;
        } else {
            variable_flags[blockIdx.x] = shared_mem->query_variable_min;
        }
    }

    if constexpr (MINIMAL_PRINTS) {
        printf("Thread %d: Simulation complete\n", threadIdx.x);
    }
}
