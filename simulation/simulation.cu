// Created by andwh on 24/10/2024.

#include "simulation.cuh"
#include <cmath>

#include "state/SharedModelState.cuh"
#include "state/SharedRunState.cuh"
#include "../main.cuh"

#define NUM_RUNS 6
#define TIME_BOUND 10.0
#define MAX_VARIABLES 20



__device__ double evaluate_expression(const expr* e, BlockSimulationState* block_state) {
    if(e == nullptr) {
        printf("Warning: Null expression in evaluate_expression\n");
        return 0.0;
    }

    // Get operand using our helper
    int op = fetch_expr_operand(e);

    if(op == expr::pn_compiled_ee) {
        if constexpr (VERBOSE) {
            printf("DEBUG: Evaluating Polish notation expression of length %d\n",
                   e->length); // Direct access ok since we're just reading length once
        }

        // Create a stack for evaluation
        __shared__ double value_stack[64]; // TODO: We can probably get the exact number from the optimizer, and pass it in as a parameter
        int stack_top = 0;

        // Evaluate the Polish notation expression
        // Since e[i] are contiguous (we use polish notation), we can access them directly
        for(int i = 1; i < e->length; i++) {  // Start from 1 to skip the header
            const expr& current = e[i];
            int current_op = fetch_expr_operand(&current);

            switch(current_op) {
                case expr::literal_ee:
                    value_stack[stack_top++] = fetch_expr_value(&current); // stack_top++ increments the stack_top after adding the value
                    break;

                case expr::clock_variable_ee: {
                    int var_id = current.variable_id;  // Union access, single read
                    value_stack[stack_top++] =
                        block_state->shared->variables[var_id].value;
                    break;
                }

                case expr::plus_ee: {
                    double b = value_stack[--stack_top]; // --stack_top first decrements then accesses
                    double a = value_stack[--stack_top];
                    value_stack[stack_top++] = a + b;
                    break;
                }

                case expr::minus_ee: {
                    double b = value_stack[--stack_top];
                    double a = value_stack[--stack_top];
                    value_stack[stack_top++] = a - b;
                    break;
                }

                case expr::multiply_ee: {
                    double b = value_stack[--stack_top];
                    double a = value_stack[--stack_top];
                    value_stack[stack_top++] = a * b;
                    break;
                }

                // TODO: Add more operators as needed

                default:
                    printf("Warning: Unknown operator %d in PN expression\n", current_op);
                    break;
            }
        }
        return value_stack[stack_top - 1];
    }


    // Handle non-PN expressions
    if constexpr (VERBOSE) {
        printf("DEBUG: Evaluating non-pn expression with operator %d\n", op);
    }

    switch(op) {
        case expr::literal_ee:
            return fetch_expr_value(e);

        case expr::clock_variable_ee: {
            int var_id = e->variable_id;  // Union access, single read
            if(var_id < MAX_VARIABLES) {
                return block_state->shared->variables[var_id].value;
            }
            printf("Warning: Invalid variable ID %d in expression\n", var_id);
            return 0.0;
        }

        case expr::plus_ee: {
            const expr* left = fetch_expr(e->left);
            const expr* right = fetch_expr(e->right);
            if(left && right) {
                double left_val = evaluate_expression(left, block_state);
                double right_val = evaluate_expression(right, block_state);
                if constexpr (VERBOSE) {
                    printf("DEBUG: Plus operation: %f + %f = %f\n",
                           left_val, right_val, left_val + right_val);
                }
                return left_val + right_val;
            }
            break;
        }

        case expr::minus_ee: {
            const expr* left = fetch_expr(e->left);
            const expr* right = fetch_expr(e->right);
            if(left && right) {
                double left_val = evaluate_expression(left, block_state);
                double right_val = evaluate_expression(right, block_state);
                return left_val - right_val;
            }
            break;
        }

        case expr::multiply_ee: {
            const expr* left = fetch_expr(e->left);
            const expr* right = fetch_expr(e->right);
            if(left && right) {
                double left_val = evaluate_expression(left, block_state);
                double right_val = evaluate_expression(right, block_state);
                return left_val * right_val;
            }
            break;
        }

        case expr::division_ee: {
            const expr* left = fetch_expr(e->left);
            const expr* right = fetch_expr(e->right);
            if(left && right) {
                double left_val = evaluate_expression(left, block_state);
                double right_val = evaluate_expression(right, block_state);
                if(right_val == 0.0) {
                    printf("Warning: Division by zero\n");
                    return 0.0;
                }
                return left_val / right_val;
            }
            break;
        }

        case expr::pn_skips_ee:
            if(e->left) {
                return evaluate_expression(fetch_expr(e->left), block_state);
            }
            break;

        default:
            printf("Warning: Unhandled operator %d in expression\n", op);
            break;
    }

    return 0.0;
}





__device__ void take_transition(ComponentState* my_state,
                              SharedBlockMemory* shared,
                              SharedModelState* model,
                              BlockSimulationState* block_state) {
    if(my_state->num_enabled_edges == 0) {
        printf("Thread %d: No enabled edges to take\n", threadIdx.x);
        return;
    }

    // Select random edge from enabled ones
    int selected_idx;
    if(my_state->num_enabled_edges == 1) {
        selected_idx = my_state->enabled_edges[0];
        if constexpr (VERBOSE) {
            printf("Thread %d: Only one enabled edge (%d), selecting it\n",
                   threadIdx.x, selected_idx);
        }
    } else {
        // Random selection between enabled edges
        float rand = curand_uniform(block_state->random);
        selected_idx = my_state->enabled_edges[(int)(rand * my_state->num_enabled_edges)];
        if constexpr (VERBOSE) {
            printf("Thread %d: Randomly selected edge %d from %d enabled edges (rand=%f)\n",
                   threadIdx.x, selected_idx, my_state->num_enabled_edges, rand);
        }
    }

    // Get the selected edge
    const EdgeInfo& edge = model->edges[my_state->current_node->first_edge_index + selected_idx];
    if constexpr (VERBOSE) {
        printf("Thread %d: Taking transition from node %d to node %d\n",
               threadIdx.x, my_state->current_node->id, edge.dest_node_id);
    }

    // If this edge has a positive channel (broadcast sender)
    if(edge.channel > 0) {
        int channel_abs = abs(edge.channel);

        // This component needs to signal the broadcast, as its channel is '!-labelled'
        shared->channel_active[channel_abs] = true;
        shared->channel_sender[channel_abs] = my_state->component_id;

        // __syncthreads(); // Wait for all threads to see broadcast. We don't need this
    }


    // Apply updates if any
    for(int i = 0; i < edge.num_updates; i++) {
        const UpdateInfo& update = model->updates[edge.updates_start_index + i];
        int var_id = update.variable_id;

        // Evaluate update expression
        double new_value = evaluate_expression(update.expression, block_state);
        if constexpr (VERBOSE) {
            printf("Thread %d: Update %d - Setting var_%d (%s) from %f to %f\n",
                   threadIdx.x, i, var_id,
                   update.kind == VariableKind::CLOCK ? "clock" : "int",
                   shared->variables[var_id].value,
                   new_value);
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

    const NodeInfo* dest_node = nullptr;
    // Search through all level slots
    //for(int level = 0; level < model->max_nodes_per_component; level++) {
    //    int level_start = level * model->num_components;
     //   if constexpr (VERBOSE) {
    //        printf("Thread %d: Checking level %d starting at index %d\n",
    //               threadIdx.x, level, level_start);
    //    }
//
    //    for(int i = 0; i < model->num_components; i++) {
    //        const NodeInfo& node = model->nodes[level_start + i];
    //        if constexpr (VERBOSE) {
    //            printf("Thread %d:   Checking node id=%d\n", threadIdx.x, node.id);
    //        }
    //        if(node.id == edge.dest_node_id) {
    //            dest_node = &node;
    //            if constexpr (VERBOSE) {
    //                printf("Thread %d: Found destination node at level %d, index %d\n",
    //                       threadIdx.x, level, i);
    //            }
    //            break;
    //        }
    //    }
    //    if(dest_node != nullptr) break;
    //}
    // new method optimized for finding dest node
    // If multiple runs in 1 block, then we can still find relevant index by using the modulo component
    // Example: 4 runs, thread 1, 26, 51, 76 all access index 1. Aka modulo num_components
    for (int level = 0; level < model->max_nodes_per_component; level++) {
        // We know relevant nodes are stored at each level offset by the index of the component idx
        // The component idx is equivalent to threadIdx
        const NodeInfo& node = model->nodes[level * model->num_components + threadIdx.x];
        if(node.id == edge.dest_node_id) {
            dest_node = &node;
            if constexpr (VERBOSE) {
                printf("Thread %d: Found destination node at level %d, index %d\n",
                       threadIdx.x, level, (threadIdx.x));
            }
            break;
        }

    }


    if(dest_node == nullptr) {
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


__device__ bool check_edge_enabled(const EdgeInfo& edge,
                                 const SharedBlockMemory* shared,
                                 SharedModelState* model,
                                 BlockSimulationState* block_state, bool is_broadcast_sync) {
    if constexpr (VERBOSE) {
        printf("\nThread %d: Checking edge %d->%d with %d guards\n",
               threadIdx.x, edge.source_node_id, edge.dest_node_id, edge.num_guards);
    }

    // Only reject negative channels if not part of broadcast sync
    if(edge.channel < 0 && !is_broadcast_sync) {
        if constexpr (VERBOSE) {
            printf("Thread %d: Is a !-labelled channel, disabled until synchronisation\n", threadIdx.x);
        }
        return false;
    }

    // Check all guards on the edge
    for(int i = 0; i < edge.num_guards; i++) {
        const GuardInfo& guard = model->guards[edge.guards_start_index + i];

        if(guard.uses_variable) {
            int var_id = guard.var_info.variable_id;
            double var_value = shared->variables[var_id].value;
            double bound = evaluate_expression(guard.expression, block_state);
            if constexpr (VERBOSE) {
                printf("  Guard %d: var_%d (%s) = %f %s %f\n",
                       i, var_id,
                       guard.var_info.type == VariableKind::CLOCK ? "clock" : "int",
                       var_value,
                       guard.operand == constraint::less_equal_c ? "<=" :
                       guard.operand == constraint::less_c ? "<" :
                       guard.operand == constraint::greater_equal_c ? ">=" :
                       guard.operand == constraint::greater_c ? ">" : "?",
                       bound);
            }

            bool satisfied = false;
            switch(guard.operand) {
                case constraint::less_c:
                    satisfied = var_value < bound; break;
                case constraint::less_equal_c:
                    satisfied = var_value <= bound; break;
                case constraint::greater_c:
                    satisfied = var_value > bound; break;
                case constraint::greater_equal_c:
                    satisfied = var_value >= bound; break;
                default:
                    printf("  Warning: Unknown operator %d\n", guard.operand);
                    return false;
            }

            if(!satisfied) {
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

__device__ void check_enabled_edges(ComponentState* my_state,
                                  SharedBlockMemory* shared,
                                  SharedModelState* model,
                                  BlockSimulationState* block_state,
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

    const NodeInfo& node = *my_state->current_node;
    my_state->num_enabled_edges = 0;  // Reset counter

    // Check each outgoing edge
    for(int i = 0; i < node.num_edges; i++) {
        const EdgeInfo& edge = model->edges[node.first_edge_index + i];
        if(check_edge_enabled(edge, shared, model, block_state, false)) {
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




__device__ void check_cuda_error(const char* location) {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error at %s: %s\n", location, cudaGetErrorString(error));
    }
}

#define CHECK_ERROR(loc) check_cuda_error(loc)

__device__ void compute_possible_delay(
    ComponentState* my_state,
    SharedBlockMemory* shared,
    SharedModelState* model,
    BlockSimulationState* block_state, int num_vars)
{
    // First check for active broadcasts
    // if(shared->channel_sender[my_state->component_id]) { We don't need this
    for(int ch = 0; ch < MAX_CHANNELS; ch++) {
        if(shared->channel_active[ch] &&
           shared->channel_sender[ch] != my_state->component_id) {  // This is the correct check

            // Collect all enabled receiving edges
            my_state->num_enabled_edges = 0;
            const NodeInfo* current_node = my_state->current_node;

            for(int e = 0; e < current_node->num_edges; e++) {
                const EdgeInfo& edge = model->edges[current_node->first_edge_index + e];
                if(edge.channel == -ch &&
                   check_edge_enabled(edge, shared, model, block_state, true)) {
                    // Edges is enabled, therefore we add it to the enabled edges list, to be used inside take_transition
                    my_state->enabled_edges[my_state->num_enabled_edges++] = e;
                   }
            }

            // If we found any enabled receiving edges, we just randomly select one inside take_transition
            if(my_state->num_enabled_edges > 0) {
                if constexpr (VERBOSE) {
                    printf("Found %d enabled receiving edges for channel %d.\n",
                           my_state->num_enabled_edges, ch);
                }
                take_transition(my_state, shared, model, block_state);
            }
           }
    }

    // }

    __syncthreads();

    // Only a single thread needs to reset synchronisation flags in shared memory
    if(threadIdx.x == 0) {
        for(int ch = 0; ch < MAX_CHANNELS; ch++) {
            shared->channel_active[ch] = false;
            shared->channel_sender[ch] = -1;
        }
    }

    __syncthreads();


    const NodeInfo& node = *my_state->current_node;
    if constexpr (VERBOSE) {
        printf("Thread %d: Processing node %d with %d invariants\n",
               threadIdx.x, node.id, node.num_invariants);
    }

    double min_delay = 0.0;
    double max_delay = DBL_MAX;
    bool is_bounded = false;

    __syncthreads();


    if constexpr (VERBOSE || true) {
        printf("Node idx %d has type: %d \n", node.id, node.type);
    }
    // Node types with 3 (Urgent) or 4 (Comitted) need to return 0 as their delay
    if (node.type > 2) {
        if constexpr (VERBOSE || true) {
            printf("Node idx %d has type: %d, therefore it is urgent or comitted and selecting delay 0 \n", node.id, node.type);
        }
        my_state->next_delay = 0;
        my_state->has_delay = true;
        return;
    }


    // Debug current variable values
    if constexpr (VERBOSE) {
        if(threadIdx.x == 0) {
            printf("Thread %d: Current variable values:\n", threadIdx.x);
            for(int i = 0; i < num_vars; i++) {
                printf("  var[%d] = %f (rate=%d)\n", i,
                       shared->variables[i].value,
                       shared->variables[i].rate);
            }
        }
    }

    // Process invariants
    for(int i = 0; i < node.num_invariants; i++) {
        const GuardInfo& inv = model->invariants[node.first_invariant_index + i];

        if(inv.uses_variable) {
            int var_id = inv.var_info.variable_id;
            if(var_id >= num_vars) {
                printf("Thread %d: Invalid variable ID %d\n", threadIdx.x, var_id);
                continue;
            }

            auto& var = shared->variables[var_id];
            double current_val = var.value;

            // Set rate to 1 for clocks
            if(inv.var_info.type == VariableKind::CLOCK) { // TODO: fetch the rate from the model. Used for exponential distribution etc.
                var.rate = 1;
            }

            // Evaluate bound expression
            double bound = evaluate_expression(inv.expression, block_state);
            if constexpr (VERBOSE) {
                printf("Thread %d: Clock %d invariant: current=%f, bound=%f, rate=%d\n",
                       threadIdx.x, var_id, current_val, bound, var.rate); // TODO: remove rate from var. Rates are dependent on the location
            }
            // Only handle upper bounds
            if(inv.operand == constraint::less_c ||
               inv.operand == constraint::less_equal_c) {

                if(var.rate > 0) {  // Only if clock increases
                    double time_to_bound = (bound - current_val) / var.rate;

                    // Add small epsilon for strict inequality
                    if(inv.operand == constraint::less_c) {
                        time_to_bound -= 1e-6;
                    }
                    if constexpr (VERBOSE) {
                        printf("Thread %d: Computed time_to_bound=%f\n",
                               threadIdx.x, time_to_bound);
                    }

                    if(time_to_bound >= 0) {
                        max_delay = min(max_delay, time_to_bound); // TODO: remove time_to_bound, as it is not part of the semantics
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
    if(is_bounded) {
        double rand = curand_uniform(block_state->random);
        my_state->next_delay = min_delay + (max_delay - min_delay) * rand;
        my_state->has_delay = true;
        if constexpr (VERBOSE) {
            printf("Thread %d: Sampled delay %f in [%f, %f] (rand=%f)\n",
                   threadIdx.x, my_state->next_delay, min_delay, max_delay, rand);
        }
    } else {

        double rate = 1.0; // Default rate if no rate is specified on the node
        if(node.lambda != nullptr) {
            rate = evaluate_expression(node.lambda, block_state);
        }

        // Sample from the exponential distribution
        double rand = curand_uniform_double(block_state->random);
        my_state->next_delay = -__log2f(rand) / rate; // Fastest log, but not as accurate. We consider it fine because we are doing statistical sampling
        my_state->has_delay = true;
        if constexpr (VERBOSE) {
            printf("Thread %d: No delay bounds, sampled %f using exponential distribution with rate %f\n", threadIdx.x, my_state->next_delay, rate);
        }
        my_state->has_delay = true;
    }
}





__device__ double find_minimum_delay(
    ComponentState* my_state,
    SharedBlockMemory* shared,
    SharedModelState* model,
    BlockSimulationState* block_state,
    int num_components)
{
    __shared__ double delays[MAX_COMPONENTS];  // Only need MAX_COMPONENTS slots, not full warp size
    __shared__ int component_indices[MAX_COMPONENTS];

    // Initialize only for active threads (components)
    if(threadIdx.x < num_components) {
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
        if(threadIdx.x == 0) {
            printf("Initial delays: ");
            for(int i = 0; i < num_components; i++) {  // Only print actual components
                printf("[%d]=%f ", i, delays[i]);
            }
            printf("\n");
        }
        __syncthreads();
    }
    // Find minimum - only compare actual components
    for(int stride = (num_components + 1)/2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride && threadIdx.x < num_components) {
            int compare_idx = threadIdx.x + stride;
            if(compare_idx < num_components) {  // Only compare if within valid components
                if constexpr (VERBOSE) {
                    printf("Thread %d comparing with position %d: %f vs %f\n",
                           threadIdx.x, compare_idx, delays[threadIdx.x],
                           delays[compare_idx]);
                }

                if(delays[compare_idx] < delays[threadIdx.x]) {
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

    // Rest of the function same as before, but only thread 0 processes result
    double min_delay = delays[0];
    if(threadIdx.x == 0) {
        if(min_delay < DBL_MAX) {
            if constexpr (VERBOSE) {
                printf("\nFinal result:\n");
                printf("  Minimum delay: %f\n", min_delay);
                printf("  Winning component: %d\n", component_indices[0]);
                printf("\nUpdating clocks:\n");
            }

            for(int i = 0; i < MAX_VARIABLES; i++) {
                if(shared->variables[i].kind == VariableKind::CLOCK) {
                    double old_value = shared->variables[i].value;
                    shared->variables[i].rate = 1;
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
    if(threadIdx.x < num_components) {  // Only check for actual components
        // is_race_winner = (min_delay < DBL_MAX &&
        //                  component_indices[0] == my_state->component_id);
        // if(is_race_winner) {
        //     printf("\nThread %d (component %d) won race with delay %f\n",
        //            threadIdx.x, my_state->component_id, min_delay);
        // }

        bool is_race_winner = (min_delay < DBL_MAX &&
                      component_indices[0] == my_state->component_id);

        if(is_race_winner) {
            check_enabled_edges(my_state, shared, model, block_state, is_race_winner);
            take_transition(my_state, shared, model, block_state);
        }
    }

    // Check enabled edges for winning component
    // if(threadIdx.x < num_components) {  // Only process for actual components
    //     check_enabled_edges(my_state, shared, model, block_state, is_race_winner);
    // }
    __syncthreads();

    return min_delay;
}

__global__ void simulation_kernel(SharedModelState* model, bool* results,
                                int runs_per_block, float time_bound, VariableKind* kinds, int num_vars) {
    if constexpr (VERBOSE) {
        if(threadIdx.x == 0) {
            printf("Starting kernel: block=%d, thread=%d\n",
                   blockIdx.x, threadIdx.x);
        }
    }
        CHECK_ERROR("kernel start");
    if constexpr (VERBOSE) {
        if(threadIdx.x == 0) {
            printf("Number of variables: %d\n", num_vars);
        }

        // Verify model pointer
        if(model == nullptr) {
            printf("Thread %d: NULL model pointer!\n", threadIdx.x);
            return;
        }
    }

    __shared__ SharedBlockMemory shared_mem;
    __shared__ ComponentState components[MAX_COMPONENTS];
    __shared__ curandState rng_states[MAX_COMPONENTS];
    if constexpr (VERBOSE) {
        if(threadIdx.x < model->num_components) {  // Only debug print for actual components
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


    if (threadIdx.x == 0) { // TODO: remove. Double with the init function.
        // Initialize variables with default values
        for(int i = 0; i < MAX_VARIABLES; i++) {
            shared_mem.variables[i].value = 0.0;
            shared_mem.variables[i].rate = 0;  // Will be set when needed based on guards
            shared_mem.variables[i].kind = VariableKind::INT;  // Default
            shared_mem.variables[i].last_writer = -1;
        }

        // For debug: print all variables and their initial values. TODO: remove later
        if constexpr (VERBOSE) {
            if(threadIdx.x == 0) {
                printf("\nInitializing variables from model invariants:\n");
                for(int comp = 0; comp < model->num_components; comp++) {
                    const NodeInfo& node = model->nodes[comp];
                    printf("Component %d has %d invariants starting at index %d\n",
                           comp, node.num_invariants, node.first_invariant_index);

                    for(int i = 0; i < node.num_invariants; i++) {
                        const GuardInfo& inv = model->invariants[node.first_invariant_index + i];
                        printf("  Invariant %d:\n"
                               "    Uses variable: %d\n"
                               "    Variable ID: %d\n"
                               "    Initial value: %f\n",
                               i, inv.uses_variable,
                               inv.uses_variable ? inv.var_info.variable_id : -1,
                               inv.var_info.initial_value != 0.0 ? inv.var_info.initial_value : 0.0);
                    }
                }
            }
        }


    }

    __syncthreads();

    CHECK_ERROR("after shared memory declaration");

    // Debug model access
    if constexpr (VERBOSE) {
        if(threadIdx.x == 0) {
            printf("Thread %d: Attempting to access model, num_components=%d\n",
                   threadIdx.x, model->num_components);
        }
    }

    CHECK_ERROR("after model access");


    // Setup block state
    BlockSimulationState block_state;
    block_state.model = model;
    block_state.shared = &shared_mem;
    block_state.my_component = &components[threadIdx.x];
    if constexpr (VERBOSE) {
        if(threadIdx.x == 0) {
            printf("Thread %d: Block state setup complete\n", threadIdx.x);
        }
    }
    CHECK_ERROR("after block state setup");

    // Initialize RNG
    int sim_id = blockIdx.x * runs_per_block;
    int comp_id = threadIdx.x;
    curand_init(1234 + sim_id * blockDim.x + comp_id, 0, 0,
                &rng_states[threadIdx.x]);
    block_state.random = &rng_states[threadIdx.x];

    // printf("Thread %d: RNG initialized\n", threadIdx.x);
    CHECK_ERROR("after RNG init");

    // Initialize shared state
    if (threadIdx.x == 0) {
        if constexpr (VERBOSE) {
            printf("Block %d: Initializing shared memory\n", blockIdx.x);
        }
        SharedBlockMemory::init(&shared_mem, sim_id);

        for (int i = 0; i < num_vars; i++) {
            if constexpr (VERBOSE) {
                printf("Setting variable %d to kind %d\n", i, kinds[i]);
            }
            shared_mem.variables[i].value = model->initial_var_values[i];
            if(kinds[i] == VariableKind::CLOCK) {
                shared_mem.variables[i].kind = VariableKind::CLOCK;
            }
            else if(kinds[i] == VariableKind::INT) {
                // Not sure whether we need this case yet
            }
        }
    }
    __syncthreads();
    CHECK_ERROR("after shared memory init");


    // Initialize component state
    if(threadIdx.x >= model->num_components) {
        printf("Thread %d: Exiting - thread ID exceeds number of components\n",
               threadIdx.x);
        return;
    }

    ComponentState* my_state = block_state.my_component;
    my_state->component_id = comp_id;
    my_state->current_node = &model->nodes[comp_id];
    my_state->has_delay = false;
    if constexpr (VERBOSE) {
        printf("Thread %d: Component initialized, node_id=%d\n",
               threadIdx.x, my_state->current_node->id);
    }
    CHECK_ERROR("after component init");

    // Main simulation loop
    while(shared_mem.global_time < time_bound) {
        if constexpr (VERBOSE) {
            printf("Thread %d: Time=%f\n", threadIdx.x, shared_mem.global_time);
        }
        compute_possible_delay(my_state, &shared_mem, model, &block_state, num_vars);
        CHECK_ERROR("after compute delay");
        __syncthreads();

        double min_delay = find_minimum_delay(
    block_state.my_component,  // ComponentState*
    &shared_mem,              // SharedBlockMemory*
    model,                    // SharedModelState*
    &block_state,            // BlockSimulationState*
    model->num_components    // int num_components
);
        CHECK_ERROR("after find minimum");
        if constexpr (VERBOSE) {
            printf("Thread %d: Minimum delay = %f\n", threadIdx.x, min_delay);
        }
        if(threadIdx.x == 0) {
            shared_mem.global_time += min_delay;
            if constexpr (VERBOSE) {
                printf("Block %d: Advanced time to %f\n",
                       blockIdx.x, shared_mem.global_time);
            }
        }
        __syncthreads();
    }

    printf("Thread %d: Simulation complete\n", threadIdx.x);
}

void simulation::run_statistical_model_checking(SharedModelState* model, float confidence, float precision, VariableKind* kinds, int num_vars) {


   int total_runs = 1;
    if constexpr (VERBOSE) {
        cout << "total_runs = " << total_runs << endl;
    }
   // Validate parameters
   if(model == nullptr) {
       cout << "Error: NULL model pointer" << endl;
       return;
   }

   // Get device properties and validate configuration
   cudaDeviceProp deviceProp;
   cudaError_t error = cudaGetDeviceProperties(&deviceProp, 0);
   if(error != cudaSuccess) {
       cout << "Error getting device properties: " << cudaGetErrorString(error) << endl;
       return;
   }



   // Adjust threads to be multiple of warp size
   int warp_size = deviceProp.warpSize;
   int threads_per_block = ((2 + warp_size - 1) / warp_size) * warp_size; // Round up to nearest warp
   int runs_per_block = 1;
   int num_blocks = 1;

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
   if(shared_mem_per_block > deviceProp.sharedMemPerBlock) {
       cout << "Error: Required shared memory (" << shared_mem_per_block
            << ") exceeds device capability (" << deviceProp.sharedMemPerBlock << ")" << endl;
       return;
   }
    if constexpr (VERBOSE) {
    cout << "Shared memory details:" << endl
             << "  Required: " << sizeof(SharedBlockMemory) << " bytes" << endl
             << "  Aligned: " << shared_mem_per_block << " bytes" << endl;
    }

   // Allocate and validate device results array
   bool* device_results;
   error = cudaMalloc(&device_results, total_runs * sizeof(bool));
   if(error != cudaSuccess) {
       cout << "CUDA malloc error: " << cudaGetErrorString(error) << endl;
       return;
   }
    if constexpr (VERBOSE) {
        cout << "Launch configuration validated:" << endl;
        cout << "  Blocks: " << num_blocks << endl;
        cout << "  Threads per block: " << threads_per_block << endl;
        cout << "  Shared memory per block: " << shared_mem_per_block << endl;
        cout << "  Time bound: " << TIME_BOUND << endl;
    }

   // Verify model is accessible
   SharedModelState host_model;
   error = cudaMemcpy(&host_model, model, sizeof(SharedModelState), cudaMemcpyDeviceToHost);
   if(error != cudaSuccess) {
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
void* nodes_ptr;
error = cudaMemcpy(&nodes_ptr, &(model->nodes), sizeof(void*), cudaMemcpyDeviceToHost);
if(error != cudaSuccess) {
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
if(error != cudaSuccess) {
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
if(host_model.invariants != nullptr) {
    if constexpr (VERBOSE) {
        cout << "Attempting to read invariant..." << endl;
    }
    GuardInfo test_guard;
    error = cudaMemcpy(&test_guard, host_model.invariants, sizeof(GuardInfo),
                       cudaMemcpyDeviceToHost);
    if(error != cudaSuccess) {
        cout << "Error reading invariant: " << cudaGetErrorString(error) << endl;
        cudaFree(device_results);
        return;
    }
    if constexpr (VERBOSE) {
        cout << "First invariant verification:" << endl
             << "  Uses variable: " << test_guard.uses_variable << endl
             << "  Variable ID: " << (test_guard.uses_variable ?
                                     test_guard.var_info.variable_id : -1) << endl;
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
   if(error != cudaSuccess) {
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
   if(error != cudaSuccess) {
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
   if(error != cudaSuccess) {
       cout << "Previous error cleared: " << cudaGetErrorString(error) << endl;
   }

    VariableKind* d_kinds;
    error = cudaMalloc(&d_kinds, num_vars * sizeof(VariableKind));  // Assuming MAX_VARIABLES is defined
    if(error != cudaSuccess) {
        cout << "CUDA malloc error for kinds array: " << cudaGetErrorString(error) << endl;
        return;
    }

    error = cudaMemcpy(d_kinds, kinds, num_vars * sizeof(VariableKind), cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        cout << "Error copying kinds array: " << cudaGetErrorString(error) << endl;
        cudaFree(d_kinds);
        return;
    }
    if constexpr (VERBOSE) {
        cout << "Launching kernel..." << endl;
    }
   // Launch kernel
   simulation_kernel<<<num_blocks, threads_per_block>>>(
       model, device_results, runs_per_block, TIME_BOUND, d_kinds, num_vars);

   // Check for launch error
   error = cudaGetLastError();
   if(error != cudaSuccess) {
       cout << "Launch error: " << cudaGetErrorString(error) << endl;
       cudaFree(device_results);
       return;
   }

   // Check for execution error
   error = cudaDeviceSynchronize();
   if(error != cudaSuccess) {
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