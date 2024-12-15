//
// Created by andwh on 04/11/2024.
//

#include "shared_model_state.cuh"

#include <iostream>
#include "../../main.cuh"

#include "../../automata_parser/abstract_parser.h"

// Add this before init_shared_model_state
std::vector<expr *> allocated_expressions; // Global to track allocations

expr *copy_expression_to_device(const expr *host_expr) {
    cudaError_t error;
    if constexpr (VERBOSE) {
        printf("Copying expression with operand %d\n",
               host_expr ? host_expr->operand : -1);
    }
    if (host_expr == nullptr) {
        if constexpr (VERBOSE) {
            printf("Null expression, returning nullptr\n");
        }
        return nullptr;
    }

    // Add safety check
    if (reinterpret_cast<uintptr_t>(host_expr) < 1000) {
        printf("Invalid pointer detected: %p\n", static_cast<const void *>(host_expr));
        return nullptr;
    }

    // Check if this is a Polish notation expression
    if (host_expr->operand == expr::pn_compiled_ee) {
        int array_size = host_expr->length;
        if constexpr (VERBOSE) {
            printf("Copying Polish notation expression array of size %d\n", array_size);
        }
        // Allocate device memory for the entire array at once
        expr *device_expr_array;
        error = cudaMalloc(&device_expr_array, array_size * sizeof(expr));
        if (error != cudaSuccess) {
            const std::string error_msg =
                    "CUDA malloc failed in copy_expression_to_device: " +
                    std::string(cudaGetErrorString(error));

            cudaFree(device_expr_array);
            throw std::runtime_error(error_msg);
        }
        allocated_expressions.push_back(device_expr_array);

        // Copy the entire array
        error = cudaMemcpy(device_expr_array, host_expr, array_size * sizeof(expr),
                           cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            const std::string error_msg =
                    "CUDA mem copy failed in copy_expression_to_device: " +
                    std::string(cudaGetErrorString(error));
            cudaFree(device_expr_array);
            throw std::runtime_error(error_msg);
        }
        if constexpr (VERBOSE) {
            printf("Copied Polish notation array to device at %p\n",
                   static_cast<void *>(device_expr_array));
        }

        // Verify first element
        expr verify_expr;
        error = cudaMemcpy(&verify_expr, device_expr_array, sizeof(expr),
                           cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            const std::string error_msg =
                    "CUDA mem copy failed in copy_expression_to_device: " +
                    std::string(cudaGetErrorString(error));
            cudaFree(device_expr_array);
            throw std::runtime_error(error_msg);
        }
        if constexpr (VERBOSE) {
            printf("Verified first element - operand=%d, length=%d\n",
                   verify_expr.operand, verify_expr.length);
        }

        return device_expr_array;
    }


    // If it's not a PN expression, we need to handle it differently
    // Allocate device memory for this node
    expr *device_expr;
    error = cudaMalloc(&device_expr, sizeof(expr));
    if (error != cudaSuccess) {
        const std::string error_msg =
                "CUDA malloc failed in copy_expression_to_device: " +
                std::string(cudaGetErrorString(error));
        cudaFree(device_expr);
        throw std::runtime_error(error_msg);
    }
    if constexpr (VERBOSE) {
        printf("Allocated device memory at %p for operand %d\n",
               static_cast<void *>(device_expr), host_expr->operand);
    }
    allocated_expressions.push_back(device_expr);

    // Create temporary host copy
    expr temp_expr;
    if constexpr (VERBOSE) {
        printf("DEBUG: Original expression details: operand=%d, value=%f, variable_id=%d\n",
               host_expr->operand, host_expr->value, host_expr->variable_id);
    }

    try {
        temp_expr.operand = host_expr->operand;
        temp_expr.value = host_expr->value;
        if constexpr (VERBOSE) {
            printf("Copying children for operand %d - Left: %p, Right: %p\n",
                   host_expr->operand,
                   static_cast<const void *>(host_expr->left),
                   static_cast<const void *>(host_expr->right));
        }

        // Recursively copy left and right subtrees
        temp_expr.left = copy_expression_to_device(host_expr->left);
        temp_expr.right = copy_expression_to_device(host_expr->right);
        if constexpr (VERBOSE) {
            printf("Finished copying children for operand %d\n", host_expr->operand);
        }
    } catch (...) {
        printf("Exception in copy_expression_to_device while accessing host expression with operand %d\n",
               host_expr->operand);
        cudaFree(device_expr);
        allocated_expressions.pop_back();
        return nullptr;
    }

    // Copy the node to device
    error = cudaMemcpy(device_expr, &temp_expr, sizeof(expr), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        const std::string error_msg =
                "CUDA mem copy failed in copy_expression_to_device: " +
                std::string(cudaGetErrorString(error));
        cudaFree(device_expr);
        throw std::runtime_error(error_msg);
    }
    if constexpr (VERBOSE) {
        printf("DEBUG: Copy complete - device_expr=%p\n", static_cast<void *>(device_expr));
    }
    // Verify it's readable
    expr verify_expr;
    error = cudaMemcpy(&verify_expr, device_expr, sizeof(expr), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        const std::string error_msg =
                "CUDA mem copy failed in copy_expression_to_device: " +
                std::string(cudaGetErrorString(error));
        throw std::runtime_error(error_msg);
    }
    if constexpr (VERBOSE) {
        printf("DEBUG: Verification read - operand=%d\n", verify_expr.operand);
    }
    return device_expr;
}


SharedModelState *init_shared_model_state(
    const network *cpu_network,
    const std::unordered_map<int, int> &node_subsystems_map,
    const std::unordered_map<int, std::list<edge> > &node_edge_map,
    const std::unordered_map<int, node *> &node_map,
    const std::unordered_map<int, VariableTrackingVisitor::VariableUsage> &variable_registry,
    const abstract_parser *parser, const int num_vars) {
    if constexpr (VERBOSE) {
        cout << "\nInitializing SharedModelState:" << endl;
        cout << "Component mapping:" << endl;
        for (const auto &pair: node_subsystems_map) {
            cout << "  Node " << pair.first << " -> Component " << pair.second << endl;
        }
    }

    // First organize nodes by component. Vectors of Components as vectors containing nodes
    std::vector<std::vector<std::pair<int, const std::list<edge> *> > > components_nodes;
    int max_component_id = -1;

    // Find number of components
    for (const auto &pair: node_subsystems_map) {
        max_component_id = std::max(max_component_id, pair.second);
    }
    components_nodes.resize(max_component_id + 1);

    // Group nodes by component
    for (const auto &pair: node_edge_map) {
        int node_id = pair.first;
        const std::list<edge> &edges = pair.second;
        int component_id = node_subsystems_map.at(node_id);
        components_nodes[component_id].push_back({node_id, &edges});
    }

    // Sort nodes in each component by ID for consistent ordering
    for (auto &component: components_nodes) {
        std::sort(component.begin(), component.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
    }

    // After grouping nodes by component:
    if constexpr (VERBOSE) {
        cout << "\nSorted nodes by component:" << endl;
        for (int i = 0; i < components_nodes.size(); i++) {
            cout << "Component " << i << " nodes: ";
            for (const auto &node_pair: components_nodes[i]) {
                cout << node_pair.first << " ";
            }
            cout << endl;
        }

        cout << "\nNodes by component:" << endl;
        for (int i = 0; i < components_nodes.size(); i++) {
            cout << "Component " << i << " has " << components_nodes[i].size() << " nodes:" << endl;
            for (const auto &node_pair: components_nodes[i]) {
                node *current_node = node_map.at(node_pair.first);
                cout << "  Node " << node_pair.first
                        << " with " << current_node->invariants.size << " invariants" << endl;

                // Print invariant details
                for (int j = 0; j < current_node->invariants.size; j++) {
                    const constraint &inv = current_node->invariants.store[j];
                    cout << "    Invariant " << j << ": uses_variable=" << inv.uses_variable;
                    if (inv.uses_variable) {
                        cout << ", var_id=" << inv.variable_id;
                    }
                    cout << endl;
                }
            }
        }
    }

    // Find max nodes per component for array sizing
    int max_nodes_per_component = 0;
    std::vector<int> component_sizes(components_nodes.size());
    for (int i = 0; i < components_nodes.size(); i++) {
        component_sizes[i] = components_nodes[i].size();
        max_nodes_per_component = std::max(max_nodes_per_component,
                                           component_sizes[i]);
    }

    // Allocate device memory for component sizes
    int *device_component_sizes;
    cudaError_t error = cudaMalloc(&device_component_sizes,
                                   components_nodes.size() * sizeof(int));
    if (error != cudaSuccess) {
        cout << "Error cudamalloc for component sizes: " << cudaGetErrorString(error) << endl;
    }
    error = cudaMemcpy(device_component_sizes, component_sizes.data(),
                       components_nodes.size() * sizeof(int),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cout << "Error copying component sizes to device: " << cudaGetErrorString(error) << endl;
    }

    // Prepare initial nodes
    int* Initial_nodes = static_cast<int *>(malloc(cpu_network->automatas.size * sizeof(int)));
    for (int i = 0; i < cpu_network->automatas.size; i++) {
        node* node = cpu_network->automatas[i];
        Initial_nodes[i] = node->id;
        if constexpr (VERBOSE) {
            printf("Initial nodes: %d\n",node->id);
        }
    }

    // Allocate device memory for initial nodes.
    int* device_initial_nodes;
    error = cudaMalloc(&device_initial_nodes,cpu_network->automatas.size * sizeof(int));
    if (error != cudaSuccess) {
        cout << "Error cudamalloc for initial nodes: " << cudaGetErrorString(error) << endl;
    }
    error = cudaMemcpy(device_initial_nodes, Initial_nodes,
                       cpu_network->automatas.size * sizeof(int),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cout << "Error copying component sizes to device: " << cudaGetErrorString(error) << endl;
    }


    std::vector<int> initial_values(num_vars);
    for (const auto &[var_id, var_usage]: variable_registry) {
        // Get initial value from network if available
        double initial_value = 0.0;
        for (int i = 0; i < cpu_network->variables.size; i++) {
            if (cpu_network->variables.store[i].id == var_id) {
                initial_value = cpu_network->variables.store[i].value;
                initial_values[var_id] = initial_value;
                if constexpr (VERBOSE) {
                    printf("Initializing variable %d with value %f (from network)\n",
                           var_id, initial_value);
                }
                break;
            }
        }
    }

    // Count total edges, guards, updates and invariants
    int total_edges = 0;
    int total_guards = 0;
    int total_updates = 0;
    int total_invariants = 0;
    for (const auto &pair: node_edge_map) {
        for (const auto &edge: pair.second) {
            total_edges++;
            total_guards += edge.guards.size;
            total_updates += edge.updates.size;
        }
        node *current_node = node_map.at(pair.first);
        total_invariants += current_node->invariants.size;
    }

    // Ensure that we have at least 1, so we don't end up with allocation issues
    total_edges = std::max(1, total_edges);
    total_guards = std::max(1, total_guards);
    total_updates = std::max(1, total_updates);
    total_invariants = std::max(1, total_invariants);


    // Allocate device memory
    const int total_node_slots = max_nodes_per_component * components_nodes.size();
    NodeInfo *device_nodes;
    EdgeInfo *device_edges;
    GuardInfo *device_guards;
    UpdateInfo *device_updates;
    GuardInfo *device_invariants;

    cudaMalloc(&device_nodes, total_node_slots * sizeof(NodeInfo));
    cudaMalloc(&device_edges, total_edges * sizeof(EdgeInfo));
    cudaMalloc(&device_guards, total_guards * sizeof(GuardInfo));
    cudaMalloc(&device_updates, total_updates * sizeof(UpdateInfo));
    cudaMalloc(&device_invariants, total_invariants * sizeof(GuardInfo));

    // Create host arrays
    std::vector<NodeInfo> host_nodes;
    std::vector<EdgeInfo> host_edges;
    std::vector<GuardInfo> host_guards;
    std::vector<UpdateInfo> host_updates;
    std::vector<GuardInfo> host_invariants;

    host_nodes.reserve(total_node_slots);
    host_edges.reserve(total_edges);
    host_guards.reserve(total_guards);
    host_updates.reserve(total_updates);
    host_invariants.reserve(total_invariants);

    int current_edge_index = 0;
    int current_guard_index = 0;
    int current_update_index = 0;
    int current_invariant_index = 0;


    // Helper function for creating variable-based guards
    auto create_variable_guard = [&](const constraint &guard) -> GuardInfo {
        if (guard.uses_variable) {
            // Handle direct variable reference (clocks)
            auto var_it = variable_registry.find(guard.variable_id);
            if (var_it != variable_registry.end()) {
                const auto &var_usage = var_it->second;

                // Get initial value from network/parser
                double initial_value = 0.0;
                for (int i = 0; i < cpu_network->variables.size; i++) {
                    if (cpu_network->variables.store[i].id == guard.variable_id) {
                        initial_value = cpu_network->variables.store[i].value;

                        initial_values[guard.variable_id] = initial_value; // Add to initial values array
                        if constexpr (VERBOSE) {
                            printf("Appending initial value %f for int variable %d\n", initial_value,
                                   guard.variable_id);
                            printf("DEBUG: Initial value of clock variable %d is %f\n",
                                   guard.variable_id, initial_value);
                        }
                        break;
                    }
                }

                VariableInfo var_info{
                    guard.variable_id,
                    var_usage.kind,
                    var_usage.name.c_str(),
                    initial_value
                };


                expr *device_expression = copy_expression_to_device(guard.expression);
                return GuardInfo(guard.operand, var_info, device_expression);
            }
        } else if (guard.value != nullptr && guard.value->operand == expr::clock_variable_ee) {
            // Handle variable reference in expression (integers)
            int var_id = guard.value->variable_id;
            auto var_it = variable_registry.find(var_id);
            if (var_it != variable_registry.end()) {
                const auto &var_usage = var_it->second;

                // Get initial value from network
                double initial_value = 0.0;
                for (int i = 0; i < cpu_network->variables.size; i++) {
                    if (cpu_network->variables.store[i].id == var_id) {
                        initial_value = cpu_network->variables.store[i].value;
                        if constexpr (VERBOSE) {
                            printf("Appending initial value %f for int variable %d\n", initial_value, var_id);
                        }
                        initial_values[var_id] = initial_value; // Add to initial values array
                        if constexpr (VERBOSE) {
                            printf("DEBUG: Initial value of integer variable %d is %f\n",
                                   var_id, initial_value);
                        }
                        break;
                    }
                }


                VariableInfo var_info{
                    var_id,
                    var_usage.kind,
                    var_usage.name.c_str(),
                    initial_value
                };

                expr *device_expression = copy_expression_to_device(guard.expression);
                return GuardInfo(guard.operand, var_info, device_expression);
            }
        }

        // Default case if no variable found
        printf("Warning: Variable not found in registry\n");
        char default_name[MAX_VAR_NAME_LENGTH];
        snprintf(default_name, MAX_VAR_NAME_LENGTH, "var_unknown");

        VariableInfo var_info{
            -1, // Invalid ID
            VariableKind::INT,
            default_name,
            0.0 // Default value
        };

        expr *device_expression = copy_expression_to_device(guard.expression);
        return GuardInfo(guard.operand, var_info, device_expression);
    };


    // Helper function for creating updates
    auto create_update = [&](const update &upd) -> UpdateInfo {
        auto var_it = variable_registry.find(upd.variable_id);
        if (var_it != variable_registry.end()) {
            const auto &var_usage = var_it->second;
            expr *device_expression = copy_expression_to_device(upd.expression);

            return UpdateInfo{
                upd.variable_id,
                device_expression,
                var_usage.kind
            };
        } else {
            printf("Warning: Variable ID %d not found in registry for update\n", upd.variable_id);
            expr *device_expression = copy_expression_to_device(upd.expression);
            return UpdateInfo{
                upd.variable_id,
                device_expression,
                VariableKind::INT // Default to INT
            };
        }
    };

    // For each node level
    for (int node_idx = 0; node_idx < max_nodes_per_component; node_idx++) {
        // For each component at this level
        for (int comp_idx = 0; comp_idx < components_nodes.size(); comp_idx++) {
            if (node_idx < components_nodes[comp_idx].size()) {
                const auto &node_pair = components_nodes[comp_idx][node_idx];
                int node_id = node_pair.first;
                const std::list<edge> &edges = *node_pair.second;
                node *current_node = node_map.at(node_id);

                // Store invariants
                int invariants_start = current_invariant_index;
                if constexpr (VERBOSE) {
                    cout << "Processing invariants for node " << node_id
                            << " starting at index " << invariants_start << endl;
                }

                for (int i = 0; i < current_node->invariants.size; i++) {
                    const constraint &inv = current_node->invariants.store[i];
                    if constexpr (VERBOSE) {
                        cout << "  Adding invariant " << i << " at index "
                                << current_invariant_index << endl;
                    }

                    // For invariants:
                    if (inv.uses_variable) {
                        if constexpr (VERBOSE) {
                            cout << "    Variable-based guard for var_id " << inv.variable_id << endl;
                        }
                        host_invariants.push_back(create_variable_guard(inv));
                    } else if (inv.value != nullptr && inv.value->operand == expr::clock_variable_ee) {
                        if constexpr (VERBOSE) {
                            cout << "    Value-based guard with integer variable " << inv.value->variable_id << endl;
                        }
                        host_invariants.push_back(create_variable_guard(inv));
                        // Modified create_variable_guard will handle this
                    } else {
                        if constexpr (VERBOSE) {
                            cout << "    Non-variable value-based guard" << endl;
                        }
                        expr *device_value = copy_expression_to_device(inv.value);
                        expr *device_expression = copy_expression_to_device(inv.expression);
                        host_invariants.push_back(GuardInfo(
                            inv.operand,
                            false,
                            device_value,
                            device_expression
                        ));
                    }
                    current_invariant_index++;
                }
                // Create node info
                NodeInfo node_info{
                    node_id,
                    current_node->type,
                    node_idx,
                    copy_expression_to_device(current_node->lamda),
                    current_edge_index,
                    static_cast<int>(edges.size()),
                    invariants_start,
                    static_cast<int>(current_node->invariants.size)
                };

                host_nodes.push_back(node_info);

                // Process edges
                for (const edge &e: edges) {
                    // Store guards
                    int guards_start = current_guard_index;
                    for (int g = 0; g < e.guards.size; g++) {
                        const constraint &guard = e.guards.store[g];

                        if (guard.uses_variable) {
                            if constexpr (VERBOSE) {
                                cout << "    Direct variable guard" << endl;
                            }

                            host_guards.push_back(create_variable_guard(guard));
                        } else if (guard.value != nullptr && guard.value->operand == expr::clock_variable_ee) {
                            if constexpr (VERBOSE) {
                                cout << "    Integer variable in expression" << endl;
                            }
                            host_guards.push_back(create_variable_guard(guard));
                            // Modified create_variable_guard will handle this
                        } else {
                            if constexpr (VERBOSE) {
                                cout << "    Non-variable guard" << endl;
                            }
                            expr *device_value = copy_expression_to_device(guard.value);
                            expr *device_expression = copy_expression_to_device(guard.expression);
                            if constexpr (VERBOSE) {
                                printf("DEBUG: Creating guard with expression ptr=%p\n",
                                       static_cast<const void *>(device_expression));
                            }
                            host_guards.push_back(GuardInfo(
                                guard.operand,
                                false,
                                device_value,
                                device_expression
                            ));
                            if constexpr (VERBOSE) {
                                printf("DEBUG: Added guard, expression ptr=%p\n",
                                       static_cast<const void *>(host_guards.back().expression));
                            }
                        }
                        current_guard_index++;
                    }

                    // Store updates
                    int updates_start = current_update_index;
                    for (int u = 0; u < e.updates.size; u++) {
                        const update &upd = e.updates.store[u];
                        host_updates.push_back(create_update(upd));
                        current_update_index++;
                    }

                    // Create edge info
                    EdgeInfo edge_info{
                        node_id,
                        e.dest->id,
                        e.channel,
                        copy_expression_to_device(e.weight),
                        e.guards.size,
                        guards_start,
                        e.updates.size,
                        updates_start
                    };
                    if constexpr (VERBOSE) {
                        printf("Creating edge on channel %d\n", e.channel);
                    }
                    host_edges.push_back(edge_info);
                    current_edge_index++;
                }
            } else {
                // Padding for components with fewer nodes
                host_nodes.push_back(NodeInfo{
                    -1, // Invalid node ID
                    node::location, // Default type
                    -1, // Invalid level
                    nullptr, // No lambda
                    -1, // No edges
                    0, // Zero edges
                    -1, // No invariants
                    0 // Zero invariants
                });
            }
        }
    }

    // Copy everything to device
    error = cudaMemcpy(device_nodes, host_nodes.data(),
                       total_node_slots * sizeof(NodeInfo),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cout << "Error copying nodes to device: " << cudaGetErrorString(error) << endl;
    }
    error = cudaMemcpy(device_edges, host_edges.data(),
                       total_edges * sizeof(EdgeInfo),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cout << "Error copying edges to device: " << cudaGetErrorString(error) << endl;
    }
    error = cudaMemcpy(device_guards, host_guards.data(),
                       total_guards * sizeof(GuardInfo),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cout << "Error copying guards to device: " << cudaGetErrorString(error) << endl;
    }
    error = cudaMemcpy(device_updates, host_updates.data(),
                       total_updates * sizeof(UpdateInfo),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cout << "Error copying updates to device: " << cudaGetErrorString(error) << endl;
    }
    error = cudaMemcpy(device_invariants, host_invariants.data(),
                       total_invariants * sizeof(GuardInfo),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cout << "Error copying invariants to device: " << cudaGetErrorString(error) << endl;
    }

    // Copy initial values to device
    int *device_initial_values;
    error = cudaMalloc(&device_initial_values, num_vars * sizeof(int));
    if (error != cudaSuccess) {
        cout << "Error performing Malloc: " << cudaGetErrorString(error) << endl;
    }
    error = cudaMemcpy(device_initial_values, initial_values.data(),
                       num_vars * sizeof(int),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cout << "Error copying initial values to device: " << cudaGetErrorString(error) << endl;
    }


    // Create and copy SharedModelState
    SharedModelState host_model{
        SYNC_SIDE_EFFECT,
        static_cast<int>(components_nodes.size()),
        max_nodes_per_component,
        device_component_sizes,
        device_initial_nodes,
        device_nodes,
        device_edges,
        device_guards,
        device_updates,
        device_invariants,
        device_initial_values
    };

    SharedModelState *device_model;
    error = cudaMalloc(&device_model, sizeof(SharedModelState));
    if (error != cudaSuccess) {
        cout << "Error copying component sizes to device: " << cudaGetErrorString(error) << endl;
    }
    error = cudaMemcpy(device_model, &host_model, sizeof(SharedModelState),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cout << "Error copying model to device: " << cudaGetErrorString(error) << endl;
    }

    return device_model;
}
