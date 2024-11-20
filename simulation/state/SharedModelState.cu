//
// Created by andwh on 04/11/2024.
//

#include "SharedModelState.cuh"

#include <iostream>


#include "../../automata_parser/abstract_parser.h"

#define MAX_NODES_PER_COMPONENT 5

class uppaal_xml_parser;

void print_node_info(const node* n, const std::string& prefix = "") {
    std::cout << prefix << "Node ID: " << n->id << " Type: " << n->type << "\n";
    std::cout << prefix << "Edges:\n";
    for(int i = 0; i < n->edges.size; i++) {
        const edge& e = n->edges[i];
        std::cout << prefix << "  -> Dest ID: " << e.dest->id
                 << " Channel: " << e.channel << "\n";
    }
}

// Add this before init_shared_model_state
std::vector<expr*> allocated_expressions;  // Global to track allocations

expr* copy_expression_to_device(const expr* host_expr) {
    printf("Copying expression with operand %d\n",
           host_expr ? host_expr->operand : -1);

    if(host_expr == nullptr) {
        printf("Null expression, returning nullptr\n");
        return nullptr;
    }

    // Add safety check
    if(reinterpret_cast<uintptr_t>(host_expr) < 1000) {
        printf("Invalid pointer detected: %p\n", static_cast<const void*>(host_expr));
        return nullptr;
    }

    // Check if this is a Polish notation expression
    if(host_expr->operand == expr::pn_compiled_ee) {
        int array_size = host_expr->length;
        printf("Copying Polish notation expression array of size %d\n", array_size);

        // Allocate device memory for the entire array at once
        expr* device_expr_array;
        cudaMalloc(&device_expr_array, array_size * sizeof(expr));
        allocated_expressions.push_back(device_expr_array);

        // Copy the entire array
        cudaMemcpy(device_expr_array, host_expr, array_size * sizeof(expr),
                   cudaMemcpyHostToDevice);

        printf("Copied Polish notation array to device at %p\n",
               static_cast<void*>(device_expr_array));

        // Verify first element
        expr verify_expr;
        cudaMemcpy(&verify_expr, device_expr_array, sizeof(expr),
                   cudaMemcpyDeviceToHost);
        printf("Verified first element - operand=%d, length=%d\n",
               verify_expr.operand, verify_expr.length);

        return device_expr_array;
    }


    // If it's not a PN expression, we need to handle it differently
    // Allocate device memory for this node
    expr* device_expr;
    cudaMalloc(&device_expr, sizeof(expr));
    printf("Allocated device memory at %p for operand %d\n",
           static_cast<void*>(device_expr), host_expr->operand);
    allocated_expressions.push_back(device_expr);

    // Create temporary host copy
    expr temp_expr;

    printf("DEBUG: Original expression details: operand=%d, value=%f, variable_id=%d\n",
       host_expr->operand, host_expr->value, host_expr->variable_id);

    try {
        temp_expr.operand = host_expr->operand;
        temp_expr.value = host_expr->value;

        printf("Copying children for operand %d - Left: %p, Right: %p\n",
               host_expr->operand,
               static_cast<const void*>(host_expr->left),
               static_cast<const void*>(host_expr->right));

        // Recursively copy left and right subtrees
        temp_expr.left = copy_expression_to_device(host_expr->left);
        temp_expr.right = copy_expression_to_device(host_expr->right);

        printf("Finished copying children for operand %d\n", host_expr->operand);
    }
    catch(...) {
        printf("Exception while accessing host expression with operand %d\n",
               host_expr->operand);
        cudaFree(device_expr);
        allocated_expressions.pop_back();
        return nullptr;
    }

    // Copy the node to device
    cudaMemcpy(device_expr, &temp_expr, sizeof(expr), cudaMemcpyHostToDevice);
    printf("DEBUG: Copy complete - device_expr=%p\n", static_cast<void*>(device_expr));

    // Verify it's readable
    expr verify_expr;
    cudaMemcpy(&verify_expr, device_expr, sizeof(expr), cudaMemcpyHostToDevice);
    printf("DEBUG: Verification read - operand=%d\n", verify_expr.operand);

    return device_expr;

}




SharedModelState* init_shared_model_state(
    const network* cpu_network,
    const std::unordered_map<int, int>& node_subsystems_map,
    const std::unordered_map<int, std::list<edge>>& node_edge_map,
    const std::unordered_map<int, node*>& node_map,
    const std::unordered_map<int, VariableTrackingVisitor::VariableUsage>& variable_registry,
    const abstract_parser* parser, const int num_vars)
{
    cout << "\nInitializing SharedModelState:" << endl;
    cout << "Component mapping:" << endl;
    for(const auto& pair : node_subsystems_map) {
        cout << "  Node " << pair.first << " -> Component " << pair.second << endl;
    }

    // First organize nodes by component
    std::vector<std::vector<std::pair<int, const std::list<edge>*>>> components_nodes;
    int max_component_id = -1;

    // Find number of components
    for(const auto& pair : node_subsystems_map) {
        max_component_id = std::max(max_component_id, pair.second);
    }
    components_nodes.resize(max_component_id + 1);

    // Group nodes by component
    for(const auto& pair : node_edge_map) {
        int node_id = pair.first;
        const std::list<edge>& edges = pair.second;
        int component_id = node_subsystems_map.at(node_id);
        components_nodes[component_id].push_back({node_id, &edges});
    }

    // Sort nodes in each component by ID for consistent ordering
    for(auto& component : components_nodes) {
        std::sort(component.begin(), component.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    cout << "\nSorted nodes by component:" << endl;
    for(int i = 0; i < components_nodes.size(); i++) {
        cout << "Component " << i << " nodes: ";
        for(const auto& node_pair : components_nodes[i]) {
            cout << node_pair.first << " ";
        }
        cout << endl;
    }

    // After grouping nodes by component:
    cout << "\nNodes by component:" << endl;
    for(int i = 0; i < components_nodes.size(); i++) {
        cout << "Component " << i << " has " << components_nodes[i].size() << " nodes:" << endl;
        for(const auto& node_pair : components_nodes[i]) {
            node* current_node = node_map.at(node_pair.first);
            cout << "  Node " << node_pair.first
                 << " with " << current_node->invariants.size << " invariants" << endl;

            // Print invariant details
            for(int j = 0; j < current_node->invariants.size; j++) {
                const constraint& inv = current_node->invariants.store[j];
                cout << "    Invariant " << j << ": uses_variable=" << inv.uses_variable;
                if(inv.uses_variable) {
                    cout << ", var_id=" << inv.variable_id;
                }
                cout << endl;
            }
        }
    }

    // Find max nodes per component for array sizing
    int max_nodes_per_component = 0;
    std::vector<int> component_sizes(components_nodes.size());
    for(int i = 0; i < components_nodes.size(); i++) {
        component_sizes[i] = components_nodes[i].size();
        max_nodes_per_component = std::max(max_nodes_per_component,
                                         component_sizes[i]);
    }

    // Allocate device memory for component sizes
    int* device_component_sizes;
    cudaMalloc(&device_component_sizes,
               components_nodes.size() * sizeof(int));
    cudaMemcpy(device_component_sizes, component_sizes.data(),
               components_nodes.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    std::vector<int> initial_values(num_vars);


    // Count total edges, guards, updates and invariants
    int total_edges = 0;
    int total_guards = 0;
    int total_updates = 0;
    int total_invariants = 0;
    for(const auto& pair : node_edge_map) {
        for(const auto& edge : pair.second) {
            total_edges++;
            total_guards += edge.guards.size;
            total_updates += edge.updates.size;
        }
        node* current_node = node_map.at(pair.first);
        total_invariants += current_node->invariants.size;
    }

    // Allocate device memory
    const int total_node_slots = max_nodes_per_component * components_nodes.size();
    NodeInfo* device_nodes;
    EdgeInfo* device_edges;
    GuardInfo* device_guards;
    UpdateInfo* device_updates;
    GuardInfo* device_invariants;

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
    auto create_variable_guard = [&](const constraint& guard) -> GuardInfo {
   if(guard.uses_variable) {
       // Handle direct variable reference (clocks)
       auto var_it = variable_registry.find(guard.variable_id);
       if(var_it != variable_registry.end()) {
           const auto& var_usage = var_it->second;

           // Get initial value from network/parser
           double initial_value = 0.0;
           for(int i = 0; i < cpu_network->variables.size; i++) {
               if(cpu_network->variables.store[i].id == guard.variable_id) {
                   initial_value = cpu_network->variables.store[i].value;

                   initial_values[guard.variable_id] = initial_value; // Add to initial values array
                   printf("Appending initial value %f for int variable %d\n", initial_value, guard.variable_id);
                   printf("DEBUG: Initial value of clock variable %d is %f\n",
                          guard.variable_id, initial_value);
                   break;
               }
           }

           VariableInfo var_info{
               guard.variable_id,
               var_usage.kind,
               var_usage.name.c_str(),
               initial_value
           };

           expr* device_expression = copy_expression_to_device(guard.expression);
           return GuardInfo(guard.operand, var_info, device_expression);
       }
   } else if(guard.value != nullptr &&
             guard.value->operand == expr::clock_variable_ee) {
       // Handle variable reference in expression (integers)
       int var_id = guard.value->variable_id;
       auto var_it = variable_registry.find(var_id);
       if(var_it != variable_registry.end()) {
           const auto& var_usage = var_it->second;

           // Get initial value from network
           double initial_value = 0.0;
           for(int i = 0; i < cpu_network->variables.size; i++) {
               if(cpu_network->variables.store[i].id == var_id) {
                   initial_value = cpu_network->variables.store[i].value;
                   printf("Appending initial value %f for int variable %d\n", initial_value, var_id);
                   initial_values[var_id] = initial_value; // Add to initial values array
                   printf("DEBUG: Initial value of integer variable %d is %f\n",
                          var_id, initial_value);
                   break;
               }
           }


           VariableInfo var_info{
               var_id,
               var_usage.kind,
               var_usage.name.c_str(),
               initial_value
           };

           expr* device_expression = copy_expression_to_device(guard.expression);
           return GuardInfo(guard.operand, var_info, device_expression);
       }
   }

   // Default case if no variable found
   printf("Warning: Variable not found in registry\n");
   char default_name[MAX_VAR_NAME_LENGTH];
   snprintf(default_name, MAX_VAR_NAME_LENGTH, "var_unknown");

   VariableInfo var_info{
       -1,  // Invalid ID
       VariableKind::INT,
       default_name,
       0.0  // Default value
   };

   expr* device_expression = copy_expression_to_device(guard.expression);
   return GuardInfo(guard.operand, var_info, device_expression);
};



    // Helper function for creating updates
    auto create_update = [&](const update& upd) -> UpdateInfo {
        auto var_it = variable_registry.find(upd.variable_id);
        if(var_it != variable_registry.end()) {
            const auto& var_usage = var_it->second;
            expr* device_expression = copy_expression_to_device(upd.expression);

            return UpdateInfo{
                upd.variable_id,
                device_expression,
                var_usage.kind
            };
        } else {
            printf("Warning: Variable ID %d not found in registry for update\n", upd.variable_id);
            expr* device_expression = copy_expression_to_device(upd.expression);
            return UpdateInfo{
                upd.variable_id,
                device_expression,
                VariableKind::INT  // Default to INT
            };
        }
    };

    // For each node level
    for(int node_idx = 0; node_idx < max_nodes_per_component; node_idx++) {
        // For each component at this level
        for(int comp_idx = 0; comp_idx < components_nodes.size(); comp_idx++) {
            if(node_idx < components_nodes[comp_idx].size()) {
                const auto& node_pair = components_nodes[comp_idx][node_idx];
                int node_id = node_pair.first;
                const std::list<edge>& edges = *node_pair.second;
                node* current_node = node_map.at(node_id);

                // Store invariants
                int invariants_start = current_invariant_index;
                cout << "Processing invariants for node " << node_id
                     << " starting at index " << invariants_start << endl;

                for(int i = 0; i < current_node->invariants.size; i++) {
                    const constraint& inv = current_node->invariants.store[i];
                    cout << "  Adding invariant " << i << " at index "
                         << current_invariant_index << endl;

                    // For invariants:
                    if(inv.uses_variable) {
                        cout << "    Variable-based guard for var_id " << inv.variable_id << endl;
                        host_invariants.push_back(create_variable_guard(inv));
                    } else if(inv.value != nullptr && inv.value->operand == expr::clock_variable_ee) {
                        cout << "    Value-based guard with integer variable " << inv.value->variable_id << endl;
                        host_invariants.push_back(create_variable_guard(inv));  // Modified create_variable_guard will handle this
                    } else {
                        cout << "    Non-variable value-based guard" << endl;
                        expr* device_value = copy_expression_to_device(inv.value);
                        expr* device_expression = copy_expression_to_device(inv.expression);
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
                for(const edge& e : edges) {
                    // Store guards
                    int guards_start = current_guard_index;
                    for(int g = 0; g < e.guards.size; g++) {
                        const constraint& guard = e.guards.store[g];

                        if(guard.uses_variable) {
                            cout << "    Direct variable guard" << endl;
                            host_guards.push_back(create_variable_guard(guard));
                        } else if(guard.value != nullptr && guard.value->operand == expr::clock_variable_ee) {
                            cout << "    Integer variable in expression" << endl;
                            host_guards.push_back(create_variable_guard(guard));  // Modified create_variable_guard will handle this
                        } else {
                            cout << "    Non-variable guard" << endl;
                            expr* device_value = copy_expression_to_device(guard.value);
                            expr* device_expression = copy_expression_to_device(guard.expression);
                            printf("DEBUG: Creating guard with expression ptr=%p\n",
                                   static_cast<const void*>(device_expression));
                            host_guards.push_back(GuardInfo(
                                guard.operand,
                                false,
                                device_value,
                                device_expression
                            ));
                            printf("DEBUG: Added guard, expression ptr=%p\n",
                                   static_cast<const void*>(host_guards.back().expression));
                        }
                        current_guard_index++;

                    }

                    // Store updates
                    int updates_start = current_update_index;
                    for(int u = 0; u < e.updates.size; u++) {
                        const update& upd = e.updates.store[u];
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
                    printf("Creating edge on channel %d\n", e.channel);
                    host_edges.push_back(edge_info);
                    current_edge_index++;
                }
            } else {
                // Padding for components with fewer nodes
                host_nodes.push_back(NodeInfo{
                    -1,                    // Invalid node ID
                    node::location,        // Default type
                    -1,                    // Invalid level
                    nullptr,               // No lambda
                    -1,                    // No edges
                    0,                     // Zero edges
                    -1,                    // No invariants
                    0                      // Zero invariants
                });
            }
        }
    }

    printf("DEBUG: About to copy guards to device, first guard expr=%p\n",
       static_cast<const void*>(host_guards[0].expression));

    // Before copying to device
    cout << "\nFinal arrays before device copy:" << endl;
    cout << "Nodes:" << endl;
    for(const auto& node : host_nodes) {
        cout << "  ID: " << node.id
             << ", invariants: " << node.num_invariants
             << " starting at " << node.first_invariant_index << endl;
    }

    cout << "Invariants:" << endl;
    for(const auto& inv : host_invariants) {
        cout << "  Uses variable: " << inv.uses_variable;
        if(inv.uses_variable) {
            cout << ", var_id: " << inv.var_info.variable_id;
            cout << ", initial_value: " << inv.var_info.initial_value;
        }
        cout << endl;
    }

    // Copy everything to device
    cudaMemcpy(device_nodes, host_nodes.data(),
               total_node_slots * sizeof(NodeInfo),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_edges, host_edges.data(),
               total_edges * sizeof(EdgeInfo),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_guards, host_guards.data(),
               total_guards * sizeof(GuardInfo),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_updates, host_updates.data(),
               total_updates * sizeof(UpdateInfo),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_invariants, host_invariants.data(),
               total_invariants * sizeof(GuardInfo),
               cudaMemcpyHostToDevice);

    // Copy initial values to device
    int* device_initial_values;
    cudaMalloc(&device_initial_values, num_vars * sizeof(int));
    cudaMemcpy(device_initial_values, initial_values.data(),
               num_vars * sizeof(int),
               cudaMemcpyHostToDevice);



    // Create and copy SharedModelState
    SharedModelState host_model{
        static_cast<int>(components_nodes.size()),
        max_nodes_per_component,
        device_component_sizes,
        device_nodes,
        device_edges,
        device_guards,
        device_updates,
        device_invariants,
        device_initial_values
    };

    SharedModelState* device_model;
    cudaMalloc(&device_model, sizeof(SharedModelState));
    cudaMemcpy(device_model, &host_model, sizeof(SharedModelState),
               cudaMemcpyHostToDevice);



    return device_model;
}








__device__ const char* variable_kind_to_string(VariableKind kind) {
    switch(kind) {
        case VariableKind::CLOCK:  return "local_clock";
        case VariableKind::INT:    return "local_int";
        default: return "unknown";
    }
}



__device__ void debug_print_expression(const expr* e, int depth = 0) {
    printf("\nDEBUG: Entering debug_print_expression with ptr=%p\n",
           static_cast<const void*>(e));

    if(depth > 10) {
        printf("[max_depth]");
        return;
    }

    if(e == nullptr) {
        printf("[null]");
        return;
    }

    // Try to read the operand value first
    printf("DEBUG: About to read operand at %p\n", static_cast<const void*>(e));
    int op = e->operand;
    printf("DEBUG: Got operand=%d\n", op);

    printf("[");

    // Print operator type - with safety check
    printf("DEBUG: Checking operand...\n");
    if(e->operand >= 0 && e->operand <= expr::pn_skips_ee) {
        switch(e->operand) {
            case expr::literal_ee:
                printf("DEBUG: Processing literal, value=%f\n", e->value);
                printf("literal=%f", e->value);
                break;
            case expr::clock_variable_ee:
                printf("DEBUG: Processing clock var, id=%d\n", e->variable_id);
                printf("var_%d", e->variable_id);
                break;
            case expr::plus_ee:
                printf("plus");
                break;
            case expr::minus_ee:
                printf("minus");
                break;
            case expr::multiply_ee:
                printf("mult");
                break;
            case expr::division_ee:
                printf("div");
                break;
            case expr::conditional_ee:
                printf("cond");
                break;
            case expr::random_ee:
                printf("random");
                break;
            default:
                printf("op_%d", e->operand);
        }
        printf(" ");
    } else {
        printf("invalid_op_%d ", e->operand);
    }

    // Print children info before recursing
    printf("DEBUG: Left=%p, Right=%p\n",
           static_cast<const void*>(e->left),
           static_cast<const void*>(e->right));

    // Print subtrees if they exist
    if(e->left != nullptr || e->right != nullptr) {
        printf("DEBUG: Processing children...\n");
        if(e->left != nullptr) {
            printf("left=");
            debug_print_expression(e->left, depth + 1);
        } else {
            printf("left=[null]");
        }

        if(e->right != nullptr) {
            printf(" right=");
            debug_print_expression(e->right, depth + 1);
        } else {
            printf(" right=[null]");
        }
    }

    // Special handling for conditional with extra safety
    if(e->operand == expr::conditional_ee) {
        printf(" else=");
        if(e->conditional_else != nullptr) {
            debug_print_expression(e->conditional_else, depth + 1);
        } else {
            printf("[null]");
        }
    }

    printf("]");
}




__global__ void verify_expressions_kernel(SharedModelState* model) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\nVerifying Model Structure with Variable Types:\n");
        printf("==========================================\n");

        printf("Number of components: %d\n", model->num_components);
        for(int comp = 0; comp < model->num_components; comp++) {
            printf("Component %d size: %d\n", comp, model->component_sizes[comp]);
        }

        for(int node_idx = 0; node_idx < model->component_sizes[0]; node_idx++) {
            printf("\nProcessing Node Level %d:\n", node_idx);

            for(int comp = 0; comp < model->num_components; comp++) {
                printf("  Checking component %d...\n", comp);

                if(node_idx < model->component_sizes[comp]) {
                    const NodeInfo& node = model->nodes[node_idx * model->num_components + comp];
                    if(node.id == -1) {
                        printf("  Skipping padding node\n");
                        continue;
                    }

                    printf("\nComponent %d, Node ID %d:\n", comp, node.id);
                    printf("  Type: %d\n", node.type);
                    printf("  Level: %d\n", node.level);
                    printf("  First edge index: %d\n", node.first_edge_index);
                    printf("  Num edges: %d\n", node.num_edges);
                    printf("  First invariant index: %d\n", node.first_invariant_index);
                    printf("  Num invariants: %d\n", node.num_invariants);

                    // Print invariants with safety checks
                    printf("  Invariants (%d):\n", node.num_invariants);
                    for(int i = 0; i < node.num_invariants; i++) {
                        printf("    Processing invariant %d...\n", i);

                        if(node.first_invariant_index + i >= 0) {
                            printf("DEBUG: Accessing invariant at index %d\n", i);
                            const GuardInfo& inv = model->invariants[node.first_invariant_index + i];
                            printf("DEBUG: Got invariant, uses_variable=%d, operand=%d\n",
                                   inv.uses_variable, inv.operand);
                            printf("DEBUG: About to print expression at %p\n",
                                   static_cast<const void*>(inv.expression));
                            printf("    Invariant %d: ", i);

                            if(inv.uses_variable) {
                                printf("%s [%s] (ID: %d) ",
                                       inv.var_info.name,
                                       inv.var_info.type == VariableKind::CLOCK ? "CLOCK" : "INT",
                                       inv.var_info.variable_id);
                            }

                            printf("operator: %d, expression: ", inv.operand);
                            debug_print_expression(inv.expression);
                            printf("\n");
                        }
                         else {
                            printf("    Invalid invariant index!\n");
                        }
                    }

                    // Print edges with safety checks
                    printf("  Processing edges...\n");
                    if(node.num_edges > 0 && node.first_edge_index >= 0) {
                        printf("  Edges (%d-%d):\n",
                               node.first_edge_index,
                               node.first_edge_index + node.num_edges - 1);

                        for(int e = 0; e < node.num_edges; e++) {
                            const EdgeInfo& edge = model->edges[node.first_edge_index + e];
                            printf("    Edge %d: %d -> %d (channel: %d)\n",
                                   node.first_edge_index + e,
                                   edge.source_node_id,
                                   edge.dest_node_id,
                                   edge.channel);

                            // Print guards with safety checks
                            if(edge.num_guards > 0 && edge.guards_start_index >= 0) {
                                printf("      Guards (%d):\n", edge.num_guards);
                                for(int g = 0; g < edge.num_guards; g++) {
                                    const GuardInfo& guard = model->guards[edge.guards_start_index + g];
                                    printf("        Guard %d: ", g);

                                    if(guard.uses_variable) {
                                        printf("%s [%s] (ID: %d) ",
                                               guard.var_info.name,
                                               guard.var_info.type == VariableKind::CLOCK ? "CLOCK" : "INT",
                                               guard.var_info.variable_id);
                                    }

                                    printf("operator: %d, ", guard.operand);
                                    if(guard.expression) {
                                        printf("expression: ");
                                        debug_print_expression(guard.expression);
                                    } else {
                                        printf("(null expression)");
                                    }
                                    printf("\n");
                                }
                            }

                            // Print updates with safety checks
                            if(edge.num_updates > 0 && edge.updates_start_index >= 0) {
                                printf("      Updates (%d):\n", edge.num_updates);
                                for(int u = 0; u < edge.num_updates; u++) {
                                    const UpdateInfo& update = model->updates[edge.updates_start_index + u];
                                    printf("        Update %d: var_%d [%s] = ",
                                           u,
                                           update.variable_id,
                                           update.kind == VariableKind::CLOCK ? "CLOCK" : "INT");

                                    if(update.expression) {
                                        debug_print_expression(update.expression);
                                    } else {
                                        printf("(null expression)");
                                    }
                                    printf("\n");
                                }
                            }
                        }
                    } else {
                        printf("  No edges or invalid edge index\n");
                    }
                } else {
                    printf("  No node at this level for this component\n");
                }
            }
        }
        printf("\n==========================================\n");
    }
}












