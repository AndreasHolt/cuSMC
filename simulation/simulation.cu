// Created by andwh on 24/10/2024.

#include "simulation.cuh"
#include <cmath>

#include "state/SharedModelState.cuh"
#include "state/SharedRunState.cuh"

#define NUM_RUNS 6
#define TIME_BOUND 20.0

#define MAX_VARIABLES 8


// Et array af locations for et specifikt component
// En funktion der mapper en værdi i det array til den relevante node

__device__ double evaluate_expression(const expr* e, BlockSimulationState* block_state) {
    if(e == nullptr) {
        printf("Warning: Null expression in evaluate_expression\n");
        return 0.0;
    }

    // Handle literals directly
    if(e->operand == expr::literal_ee) {
        return e->value;
    }

    // Handle variable references
    if(e->operand == expr::clock_variable_ee) {
        if(e->variable_id < MAX_VARIABLES) {
            return block_state->shared->variables[e->variable_id].value;
        }
        printf("Warning: Invalid variable ID %d in expression\n", e->variable_id);
        return 0.0;
    }

    // Just return the raw value for now
    // TODO: implement full expression evaluation later when basic timing works. We need to support variables i.e. x <= l, where l is not const
    printf("Warning: Non-literal expression (op=%d), using value directly\n",
           e->operand);
    return e->value;
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
        printf("Thread %d: Only one enabled edge (%d), selecting it\n",
               threadIdx.x, selected_idx);
    } else {
        // Random selection between enabled edges
        float rand = curand_uniform(block_state->random);
        selected_idx = my_state->enabled_edges[(int)(rand * my_state->num_enabled_edges)];
        printf("Thread %d: Randomly selected edge %d from %d enabled edges (rand=%f)\n",
               threadIdx.x, selected_idx, my_state->num_enabled_edges, rand);
    }

    // Get the selected edge
    const EdgeInfo& edge = model->edges[my_state->current_node->first_edge_index + selected_idx];
    printf("Thread %d: Taking transition from node %d to node %d\n",
           threadIdx.x, my_state->current_node->id, edge.dest_node_id);

    // Apply updates if any
    for(int i = 0; i < edge.num_updates; i++) {
        const UpdateInfo& update = model->updates[edge.updates_start_index + i];
        int var_id = update.variable_id;

        // Evaluate update expression
        double new_value = evaluate_expression(update.expression, block_state);

        printf("Thread %d: Update %d - Setting var_%d (%s) from %f to %f\n",
               threadIdx.x, i, var_id,
               update.kind == VariableKind::CLOCK ? "clock" : "int",
               shared->variables[var_id].value,
               new_value);

        shared->variables[var_id].value = new_value;
        shared->variables[var_id].last_writer = my_state->component_id;
    }

    // Find destination node info
    // First find node in the same level that matches destination ID
    printf("Thread %d: Searching for destination node %d (current level=%d)\n",
           threadIdx.x, edge.dest_node_id, my_state->current_node->level);

    const NodeInfo* dest_node = nullptr;
    // Search through all level slots
    for(int level = 0; level < model->max_nodes_per_component; level++) { // TODO: Optimize this later (no pre-mature optimization for now)
        int level_start = level * model->num_components;
        printf("Thread %d: Checking level %d starting at index %d\n",
               threadIdx.x, level, level_start);

        for(int i = 0; i < model->num_components; i++) {
            const NodeInfo& node = model->nodes[level_start + i];
            printf("Thread %d:   Checking node id=%d\n", threadIdx.x, node.id);
            if(node.id == edge.dest_node_id) {
                dest_node = &node;
                printf("Thread %d: Found destination node at level %d, index %d\n",
                       threadIdx.x, level, i);
                break;
            }
        }
        if(dest_node != nullptr) break;
    }


    if(dest_node == nullptr) {
        printf("Thread %d: ERROR - Could not find destination node %d!\n",
               threadIdx.x, edge.dest_node_id);
        return;
    }

    // Update current node
    my_state->current_node = dest_node;
    printf("Thread %d: Moved to new node %d\n", threadIdx.x, dest_node->id);
}

__device__ bool check_edge_enabled(const EdgeInfo& edge,
                                 const SharedBlockMemory* shared,
                                 SharedModelState* model,
                                 BlockSimulationState* block_state) {
    printf("\nThread %d: Checking edge %d->%d with %d guards\n",
           threadIdx.x, edge.source_node_id, edge.dest_node_id, edge.num_guards);

    // Check all guards on the edge
    for(int i = 0; i < edge.num_guards; i++) {
        const GuardInfo& guard = model->guards[edge.guards_start_index + i];

        if(guard.uses_variable) {
            int var_id = guard.var_info.variable_id;
            double var_value = shared->variables[var_id].value;
            double bound = evaluate_expression(guard.expression, block_state);

            printf("  Guard %d: var_%d (%s) = %f %s %f\n",
                   i, var_id,
                   guard.var_info.type == VariableKind::CLOCK ? "clock" : "int",
                   var_value,
                   guard.operand == constraint::less_equal_c ? "<=" :
                   guard.operand == constraint::less_c ? "<" :
                   guard.operand == constraint::greater_equal_c ? ">=" :
                   guard.operand == constraint::greater_c ? ">" : "?",
                   bound);

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
                printf("  Guard not satisfied - edge disabled\n");
                return false;
            }
        }
    }

    printf("  All guards satisfied - edge enabled!\n");
    return true;
}

__device__ void check_enabled_edges(ComponentState* my_state,
                                  SharedBlockMemory* shared,
                                  SharedModelState* model,
                                  BlockSimulationState* block_state,
                                  bool is_race_winner) {
    if (!is_race_winner) {
        printf("Thread %d: Skipping edge check (didn't win race)\n", threadIdx.x);
        return;
    }

    printf("\nThread %d: Checking enabled edges for node %d\n",
           threadIdx.x, my_state->current_node->id);

    const NodeInfo& node = *my_state->current_node;
    my_state->num_enabled_edges = 0;  // Reset counter

    // Check each outgoing edge
    for(int i = 0; i < node.num_edges; i++) {
        const EdgeInfo& edge = model->edges[node.first_edge_index + i];
        if(check_edge_enabled(edge, shared, model, block_state)) {
            // Store enabled edge for later selection
            my_state->enabled_edges[my_state->num_enabled_edges++] = i;
            printf("Thread %d: Edge %d is enabled (total enabled: %d)\n",
                   threadIdx.x, i, my_state->num_enabled_edges);
        }
    }

    printf("Thread %d: Found %d enabled edges\n",
           threadIdx.x, my_state->num_enabled_edges);
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
    BlockSimulationState* block_state)
{
    const NodeInfo& node = *my_state->current_node;
    printf("Thread %d: Processing node %d with %d invariants\n",
           threadIdx.x, node.id, node.num_invariants);

    double min_delay = 0.0;
    double max_delay = DBL_MAX;
    bool is_bounded = false;
    if(threadIdx.x == 0) {  // TODO: REMOVE
        for(int i = 0; i < MAX_VARIABLES; i++) {
            if(shared->variables[i].kind == VariableKind::CLOCK) {
                shared->variables[i].rate = 1;
            }
        }
    }

    __syncthreads();

    // Debug current variable values
    if(threadIdx.x == 0) {
        printf("Thread %d: Current variable values:\n", threadIdx.x);
        for(int i = 0; i < MAX_VARIABLES; i++) {
            printf("  var[%d] = %f (rate=%d)\n", i,
                   shared->variables[i].value,
                   shared->variables[i].rate);
        }
    }



    // Process invariants
    for(int i = 0; i < node.num_invariants; i++) {
        const GuardInfo& inv = model->invariants[node.first_invariant_index + i];

        if(inv.uses_variable) {
            int var_id = inv.var_info.variable_id;
            if(var_id >= MAX_VARIABLES) {
                printf("Thread %d: Invalid variable ID %d\n", threadIdx.x, var_id);
                continue;
            }

            auto& var = shared->variables[var_id];
            double current_val = var.value;

            // Set rate to 1 for clocks
            if(inv.var_info.type == VariableKind::CLOCK) {
                var.rate = 1;
            }

            // Evaluate bound expression
            double bound = evaluate_expression(inv.expression, block_state);
            printf("Thread %d: Clock %d invariant: current=%f, bound=%f, rate=%d\n",
                   threadIdx.x, var_id, current_val, bound, var.rate);

            // Only handle upper bounds
            if(inv.operand == constraint::less_c ||
               inv.operand == constraint::less_equal_c) {

                if(var.rate > 0) {  // Only if clock increases
                    double time_to_bound = (bound - current_val) / var.rate;

                    // Add small epsilon for strict inequality
                    if(inv.operand == constraint::less_c) {
                        time_to_bound -= 1e-6;
                    }

                    printf("Thread %d: Computed time_to_bound=%f\n",
                           threadIdx.x, time_to_bound);

                    if(time_to_bound >= 0) {
                        max_delay = min(max_delay, time_to_bound);
                        is_bounded = true;
                        printf("Thread %d: Updated max_delay to %f\n",
                               threadIdx.x, max_delay);
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
        printf("Thread %d: Sampled delay %f in [%f, %f] (rand=%f)\n",
               threadIdx.x, my_state->next_delay, min_delay, max_delay, rand);
    } else {
        printf("Thread %d: No delay bounds, using 1.0\n", threadIdx.x);
        my_state->next_delay = 1.0;  // Default step if no bounds
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
        printf("Thread %d (component %d): Initial delay %f\n",
               threadIdx.x, my_state->component_id, delays[threadIdx.x]);
    }
    __syncthreads();

    // Debug print initial state
    if(threadIdx.x == 0) {
        printf("Initial delays: ");
        for(int i = 0; i < num_components; i++) {  // Only print actual components
            printf("[%d]=%f ", i, delays[i]);
        }
        printf("\n");
    }
    __syncthreads();

    // Find minimum - only compare actual components
    for(int stride = (num_components + 1)/2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride && threadIdx.x < num_components) {
            int compare_idx = threadIdx.x + stride;
            if(compare_idx < num_components) {  // Only compare if within valid components
                printf("Thread %d comparing with position %d: %f vs %f\n",
                       threadIdx.x, compare_idx, delays[threadIdx.x],
                       delays[compare_idx]);

                if(delays[compare_idx] < delays[threadIdx.x]) {
                    delays[threadIdx.x] = delays[compare_idx];
                    component_indices[threadIdx.x] = component_indices[compare_idx];
                    printf("Thread %d: Updated minimum to %f from component %d\n",
                           threadIdx.x, delays[threadIdx.x],
                           component_indices[threadIdx.x]);
                }
            }
        }
        __syncthreads();
    }

    // Rest of the function same as before, but only thread 0 processes result
    double min_delay = delays[0];
    if(threadIdx.x == 0) {
        if(min_delay < DBL_MAX) {
            printf("\nFinal result:\n");
            printf("  Minimum delay: %f\n", min_delay);
            printf("  Winning component: %d\n", component_indices[0]);
            printf("\nUpdating clocks:\n");

            for(int i = 0; i < MAX_VARIABLES; i++) {
                if(shared->variables[i].kind == VariableKind::CLOCK) {
                    double old_value = shared->variables[i].value;
                    shared->variables[i].rate = 1;
                    shared->variables[i].value += min_delay;
                    printf("  Clock %d: %f -> %f (advanced by %f)\n",
                           i, old_value, shared->variables[i].value, min_delay);
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
            take_transition(my_state, shared, model, block_state);  // Add this line
        }
    }

    // Check enabled edges for winning component
    // if(threadIdx.x < num_components) {  // Only process for actual components
    //     check_enabled_edges(my_state, shared, model, block_state, is_race_winner);
    // }
    __syncthreads();

    return min_delay;
}












int get_total_runs(float confidence, float precision) {
    // confidence level = alpha, i.e. 0.05 for 95% confidence
    // precision = epsilon, i.e. 0.01 for +-1% error

    // int total_runs = (int)ceil(log(2.0/confidence)/log(2.0*precision*precision));
    // int total_runs = static_cast<int>(ceil(log(2.0 / confidence) / log(2.0 * precision * precision)));
    int total_runs = static_cast<size_t>(ceil((log(2.0) - log(confidence)) / (2*pow(precision, 2))));
    return total_runs;
}

// TODO: what if we want to spawn 50 trains? How do we do that?


__global__ void simulation_kernel(SharedModelState* model, bool* results,
                                int runs_per_block, float time_bound, VariableKind* kinds, int num_vars) {
    if(threadIdx.x == 0) {
        printf("Starting kernel: block=%d, thread=%d\n",
               blockIdx.x, threadIdx.x);
    }
    CHECK_ERROR("kernel start");
    if(threadIdx.x == 0) {
        printf("Number of variables: %d\n", num_vars);
    }

    // Verify model pointer
    if(model == nullptr) {
        printf("Thread %d: NULL model pointer!\n", threadIdx.x);
        return;
    }

    __shared__ SharedBlockMemory shared_mem;
    __shared__ ComponentState components[MAX_COMPONENTS];
    __shared__ curandState rng_states[MAX_COMPONENTS];

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


    if (threadIdx.x == 0) {
        // Initialize variables with default values
        for(int i = 0; i < MAX_VARIABLES; i++) {
            shared_mem.variables[i].value = 0.0;
            shared_mem.variables[i].rate = 0;  // Will be set when needed based on guards
            shared_mem.variables[i].kind = VariableKind::INT;  // Default
            shared_mem.variables[i].last_writer = -1;
        }

        // For debug: print all variables and their initial values. TODO: remove later
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
                           inv.uses_variable ? inv.var_info.initial_value : 0.0);
                }
            }
        }


    }

    __syncthreads();

    CHECK_ERROR("after shared memory declaration");

    // Debug model access
    if(threadIdx.x == 0) {
        printf("Thread %d: Attempting to access model, num_components=%d\n",
               threadIdx.x, model->num_components);
    }
    CHECK_ERROR("after model access");

    // Setup block state
    BlockSimulationState block_state;
    block_state.model = model;
    block_state.shared = &shared_mem;
    block_state.my_component = &components[threadIdx.x];

    if(threadIdx.x == 0) {
        printf("Thread %d: Block state setup complete\n", threadIdx.x);
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
        printf("Block %d: Initializing shared memory\n", blockIdx.x);
        SharedBlockMemory::init(&shared_mem, sim_id);

        for (int i = 0; i < num_vars; i++) {
            printf("Setting variable %d to kind %d\n", i, kinds[i]);
            if(kinds[i] == VariableKind::CLOCK) {
                shared_mem.variables[i].kind = VariableKind::CLOCK;

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

    printf("Thread %d: Component initialized, node_id=%d\n",
           threadIdx.x, my_state->current_node->id);
    CHECK_ERROR("after component init");

    // Main simulation loop
    while(shared_mem.global_time < time_bound) {
        printf("Thread %d: Time=%f\n", threadIdx.x, shared_mem.global_time);

        compute_possible_delay(my_state, &shared_mem, model, &block_state);
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
        printf("Thread %d: Minimum delay = %f\n", threadIdx.x, min_delay);

        if(threadIdx.x == 0) {
            shared_mem.global_time += min_delay;
            printf("Block %d: Advanced time to %f\n",
                   blockIdx.x, shared_mem.global_time);
        }
        __syncthreads();
    }

    printf("Thread %d: Simulation complete\n", threadIdx.x);
}




void simulation::run_statistical_model_checking(SharedModelState* model, float confidence, float precision, VariableKind* kinds, int num_vars) {


   int total_runs = 1;
   cout << "total_runs = " << total_runs << endl;

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
   cout << "Device details:" << endl
        << "  Name: " << deviceProp.name << endl
        << "  Warp size: " << warp_size << endl
        << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << endl
        << "  Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x "
        << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << endl
        << "  Adjusted threads per block: " << threads_per_block << endl;

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

   cout << "Shared memory details:" << endl
        << "  Required: " << sizeof(SharedBlockMemory) << " bytes" << endl
        << "  Aligned: " << shared_mem_per_block << " bytes" << endl;

   // Allocate and validate device results array
   bool* device_results;
   error = cudaMalloc(&device_results, total_runs * sizeof(bool));
   if(error != cudaSuccess) {
       cout << "CUDA malloc error: " << cudaGetErrorString(error) << endl;
       return;
   }

   cout << "Launch configuration validated:" << endl;
   cout << "  Blocks: " << num_blocks << endl;
   cout << "  Threads per block: " << threads_per_block << endl;
   cout << "  Shared memory per block: " << shared_mem_per_block << endl;
   cout << "  Time bound: " << TIME_BOUND << endl;

   // Verify model is accessible
   SharedModelState host_model;
   error = cudaMemcpy(&host_model, model, sizeof(SharedModelState), cudaMemcpyDeviceToHost);
   if(error != cudaSuccess) {
       cout << "Error copying model: " << cudaGetErrorString(error) << endl;
       cudaFree(device_results);
       return;
   }

    cout << "Model verified accessible with " << host_model.num_components << " components" << endl;

// Add verification here with more safety checks
cout << "\nVerifying model transfer:" << endl;
cout << "Model contents:" << endl;
cout << "  nodes pointer: " << host_model.nodes << endl;
cout << "  invariants pointer: " << host_model.invariants << endl;
cout << "  num_components: " << host_model.num_components << endl;

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
cout << "Nodes pointer verification: " << nodes_ptr << endl;

// Now try to read one node
NodeInfo test_node;
error = cudaMemcpy(&test_node, host_model.nodes, sizeof(NodeInfo), cudaMemcpyDeviceToHost);
if(error != cudaSuccess) {
    cout << "Error reading node: " << cudaGetErrorString(error) << endl;
    cudaFree(device_results);
    return;
}
cout << "First node verification:" << endl
     << "  ID: " << test_node.id << endl
     << "  First invariant index: " << test_node.first_invariant_index << endl
     << "  Num invariants: " << test_node.num_invariants << endl;

// Only check invariants if we have a valid pointer
if(host_model.invariants != nullptr) {
    cout << "Attempting to read invariant..." << endl;
    GuardInfo test_guard;
    error = cudaMemcpy(&test_guard, host_model.invariants, sizeof(GuardInfo),
                       cudaMemcpyDeviceToHost);
    if(error != cudaSuccess) {
        cout << "Error reading invariant: " << cudaGetErrorString(error) << endl;
        cudaFree(device_results);
        return;
    }
    cout << "First invariant verification:" << endl
         << "  Uses variable: " << test_guard.uses_variable << endl
         << "  Variable ID: " << (test_guard.uses_variable ?
                                 test_guard.var_info.variable_id : -1) << endl;
} else {
    cout << "No invariants pointer available" << endl;
}




   // Check each kernel parameter
   cout << "Kernel parameter validation:" << endl;
   cout << "  model pointer: " << model << endl;
   cout << "  device_results pointer: " << device_results << endl;
   cout << "  runs_per_block: " << runs_per_block << endl;
   cout << "  TIME_BOUND: " << TIME_BOUND << endl;

   // Verify model pointer is a valid device pointer
   cudaPointerAttributes modelAttrs;
   error = cudaPointerGetAttributes(&modelAttrs, model);
   if(error != cudaSuccess) {
       cout << "Error checking model pointer: " << cudaGetErrorString(error) << endl;
       cudaFree(device_results);
       return;
   }

   cout << "Model pointer properties:" << endl;
   cout << "  type: " << (modelAttrs.type == cudaMemoryTypeDevice ? "device" : "other") << endl;
   cout << "  device: " << modelAttrs.device << endl;

   // Similarly check device_results pointer
   cudaPointerAttributes resultsAttrs;
   error = cudaPointerGetAttributes(&resultsAttrs, device_results);
   if(error != cudaSuccess) {
       cout << "Error checking results pointer: " << cudaGetErrorString(error) << endl;
       cudaFree(device_results);
       return;
   }

   cout << "Results pointer properties:" << endl;
   cout << "  type: " << (resultsAttrs.type == cudaMemoryTypeDevice ? "device" : "other") << endl;
   cout << "  device: " << resultsAttrs.device << endl;

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

   cout << "Launching kernel..." << endl;
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

   cout << "Kernel completed successfully" << endl;

   // Cleanup
   cudaFree(d_kinds);
   cudaFree(device_results);
}









__global__ void findSmallestElementInArray(float *input, int input_length, float *result, int nblocks) {
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = blockDim.x * nblocks;
    for (int i = 0; i < ceil(log2f(nThreads)); i++) { //1
        if (threadid % static_cast<int>(pow(2, i+1)==0)) {
            int correspondant = min(static_cast<int>(threadid + pow(2, i)), input_length-1);
            input[threadid] = min(input[threadid], input[correspondant]);
        }
    }
    if (threadid == 0) {
        *result = input[0];
    }
}

// void testFunction () {
//     float* h_a = new float[NUM_RUNS];
//     srand( static_cast<unsigned>(time(NULL)));
//     int upper = 3500;
//     int lower = 1230;
//     for (int i = 0; i < NUM_RUNS; i++) {
//
//         h_a[i] = rand() % (upper - lower) + lower;
//         cout << h_a[i] << ", ";
//     }
//     cout << endl;
//
//     float* d_a;
//     float d_result;
//     cudaMalloc(&d_a, NUM_RUNS * sizeof(float));
//
//     cudaMemcpy(d_a, h_a, NUM_RUNS * sizeof(float), cudaMemcpyHostToDevice);
//
//     findSmallestElementInArray<<<1, 128>>>(d_a, NUM_RUNS, &d_result, 1); // 2 blocks (component size), 100 simulations but round up to 128
//
//     cudaMemcpy(h_a, d_a, NUM_RUNS * sizeof(float), cudaMemcpyDeviceToHost);
//
//     for (int i = 0; i < NUM_RUNS; i++) {
//         cout << h_a[i] << ", ";
//     }
//     cout <<endl << "Result = " << d_result << endl;
// }

void simulation::runSimulation() {
    // Problem with models, spawning new components Trains in train gate for example?
    // componentSimulation<<<NUM_COMPONENTS, 128>>>(); // 2 blocks (component size), 100 simulations but round up to 128

    // testFunction();

    // Pick delays: implement delay function
    // Find the smallest delay, and which index it has (to find component it belongs to)
    // Apply the delay
    // Pick a transition from the component that won: Pick according to the weights
    // Check whether we need to synchronize with anything when taking this transition
    // Take the transition

    // We need the state such that we can describe the run afterwards. We add our delays to it.



    cout << "test from run sim" << endl;
}
