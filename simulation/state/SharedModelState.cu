//
// Created by andwh on 04/11/2024.
//

#include "SharedModelState.cuh"

#include <iostream>

#define MAX_NODES_PER_COMPONENT 5

std::vector<expr*> allocated_expressions;  // Global to track allocations


expr* create_variable_expr(int variable_id) {
    printf("Creating variable expression for id: %d\n", variable_id);

    expr* device_expr;
    cudaError_t err = cudaMalloc(&device_expr, sizeof(expr));
    if(err != cudaSuccess) {
        printf("cudaMalloc failed in create_variable_expr: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    allocated_expressions.push_back(device_expr);

    expr temp_expr;
    memset(&temp_expr, 0, sizeof(expr));  // Initialize to zero
    temp_expr.operand = expr::clock_variable_ee;
    temp_expr.value = static_cast<double>(variable_id);
    temp_expr.left = nullptr;
    temp_expr.right = nullptr;

    err = cudaMemcpy(device_expr, &temp_expr, sizeof(expr), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        printf("cudaMemcpy failed in create_variable_expr: %s\n", cudaGetErrorString(err));
        cudaFree(device_expr);
        return nullptr;
    }

    printf("Successfully created variable expression\n");
    return device_expr;
}


void print_node_info(const node* n, const std::string& prefix = "") {
    std::cout << prefix << "Node ID: " << n->id << " Type: " << n->type << "\n";
    std::cout << prefix << "Edges:\n";
    for(int i = 0; i < n->edges.size; i++) {
        const edge& e = n->edges[i];
        std::cout << prefix << "  -> Dest ID: " << e.dest->id
                 << " Channel: " << e.channel << "\n";
    }
}





expr* copy_expression_to_device(const expr* host_expr) {
    printf("Entering copy_expression_to_device with expr: %p\n", static_cast<const void*>(host_expr));

    // Check for invalid pointer (like 0x1, 0x2, 0x3)
    if(reinterpret_cast<uintptr_t>(host_expr) < 1000) {
        printf("Skipping invalid pointer: %p\n", static_cast<const void*>(host_expr));
        return nullptr;
    }

    if(host_expr == nullptr) {
        printf("Null expression, returning nullptr\n");
        return nullptr;
    }

    try {
        // Allocate device memory for this node
        expr* device_expr;
        cudaError_t err = cudaMalloc(&device_expr, sizeof(expr));
        if(err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return nullptr;
        }
        printf("Allocated device memory at: %p\n", static_cast<void*>(device_expr));
        allocated_expressions.push_back(device_expr);

        // Create temporary host copy
        expr temp_expr;
        memset(&temp_expr, 0, sizeof(expr));  // Initialize to zero

        // Carefully copy each field
        temp_expr.operand = host_expr->operand;
        temp_expr.value = host_expr->value;
        temp_expr.left = nullptr;   // Will set these after recursion
        temp_expr.right = nullptr;

        printf("Successfully copied basic fields. Operand: %d\n", temp_expr.operand);

        // Recursively copy left and right if they exist
        if(host_expr->left != nullptr) {
            printf("Copying left subtree\n");
            temp_expr.left = copy_expression_to_device(host_expr->left);
        }

        if(host_expr->right != nullptr) {
            printf("Copying right subtree\n");
            temp_expr.right = copy_expression_to_device(host_expr->right);
        }

        // Copy to device
        err = cudaMemcpy(device_expr, &temp_expr, sizeof(expr), cudaMemcpyHostToDevice);
        if(err != cudaSuccess) {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
            cudaFree(device_expr);
            return nullptr;
        }

        printf("Successfully copied expression to device\n");
        return device_expr;
    }
    catch(...) {
        printf("Exception in copy_expression_to_device\n");
        return nullptr;
    }
}




SharedModelState* init_shared_model_state(
   const network* cpu_network,
   const std::unordered_map<int, int>& node_subsystems_map,
   const std::unordered_map<int, std::list<edge>>& node_edge_map,
   const std::unordered_map<int, node*>& node_map)
{
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

   // Count total edges, guards, updates and invariants
   int total_edges = 0;
   int total_guards = 0;
   int total_updates = 0;
   int total_invariants = 0;
   for(const auto& pair : node_edge_map) {
       // Count edges, guards and updates
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

   // For each node level
   for(int node_idx = 0; node_idx < max_nodes_per_component; node_idx++) {
       // For each component at this level
       for(int comp_idx = 0; comp_idx < components_nodes.size(); comp_idx++) {
           if(node_idx < components_nodes[comp_idx].size()) {
               const auto& node_pair = components_nodes[comp_idx][node_idx];
               int node_id = node_pair.first;
               const std::list<edge>& edges = *node_pair.second;
               node* current_node = node_map.at(node_id);


               // Store invariants with deep copy of expressions
               int invariants_start = current_invariant_index;
               for(int i = 0; i < current_node->invariants.size; i++) {
                   const constraint& inv = current_node->invariants[i];
                   printf("Processing invariant %d for node %d\n", i, node_id);
                   printf("Value pointer: %p\n", static_cast<const void*>(inv.value));
                   printf("Expression pointer: %p\n", static_cast<const void*>(inv.expression));

                   expr* device_expression = copy_expression_to_device(inv.expression);
                   expr* left_expr = nullptr;

                   if(inv.uses_variable) {
                       left_expr = create_variable_expr(inv.variable_id);
                   }

                   printf("Creating GuardInfo with: operand=%d, uses_variable=%d\n",
                          inv.operand, inv.uses_variable);

                   host_invariants.push_back(GuardInfo{
                       inv.operand,
                       inv.uses_variable,
                       left_expr,
                       device_expression
                   });

                   printf("Successfully added invariant\n");
                   current_invariant_index++;
               }

               // Create NodeInfo with edge and invariant information
               NodeInfo node_info{
                   node_id,                    // id
                   current_node->type,         // type
                   node_idx,                   // level
                   copy_expression_to_device(current_node->lamda), // Deep copy lambda
                   current_edge_index,         // first_edge_index
                   static_cast<int>(edges.size()), // num_edges
                   invariants_start,           // first_invariant_index
                   static_cast<int>(current_node->invariants.size) // num_invariants
               };
               host_nodes.push_back(node_info);

               // Add edges and their guards/updates
               for(const edge& e : edges) {
                   // Store guards with deep copy of expressions
                   int guards_start = current_guard_index;
                   for(int g = 0; g < e.guards.size; g++) {
                       const constraint& guard = e.guards[g];

                       expr* device_value = copy_expression_to_device(guard.value);
                       expr* device_expression = copy_expression_to_device(guard.expression);

                       host_guards.push_back(GuardInfo{
                           guard.operand,
                           guard.uses_variable,
                           device_value,
                           device_expression
                       });
                       current_guard_index++;
                   }

                   // Store updates with deep copy of expressions
                   int updates_start = current_update_index;
                   for(int u = 0; u < e.updates.size; u++) {
                       const update& upd = e.updates[u];
                       expr* device_expression = copy_expression_to_device(upd.expression);

                       host_updates.push_back(UpdateInfo{
                           upd.variable_id,
                           device_expression
                       });
                       current_update_index++;
                   }

                   // Create edge info
                   EdgeInfo edge_info{
                       node_id,
                       e.dest->id,
                       e.channel,
                       copy_expression_to_device(e.weight), // Deep copy weight
                       e.guards.size,
                       guards_start,
                       e.updates.size,
                       updates_start
                   };
                   host_edges.push_back(edge_info);
                   current_edge_index++;
               }
           } else {
               // Padding for components with fewer nodes
               host_nodes.push_back(NodeInfo{
                   -1,                 // id
                   node::location,     // type
                   -1,                 // level
                   nullptr,            // lambda
                   -1,                 // first_edge_index
                   0,                  // num_edges
                   -1,                 // first_invariant_index
                   0                   // num_invariants
               });
           }
       }
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

   // Create and copy SharedModelState
   SharedModelState host_model{
       static_cast<int>(components_nodes.size()),
       device_component_sizes,
       device_nodes,
       device_edges,
       device_guards,
       device_updates,
       device_invariants
   };

   SharedModelState* device_model;
   cudaMalloc(&device_model, sizeof(SharedModelState));
   cudaMemcpy(device_model, &host_model, sizeof(SharedModelState),
              cudaMemcpyHostToDevice);

   return device_model;
}






__device__ void debug_print_expression(const expr* e, int depth = 0) {
    if(depth > 10) {
        printf("[max_depth]");
        return;
    }

    if(e == nullptr) {
        printf("null");
        return;
    }

    switch(e->operand) {
        case expr::literal_ee:
            printf("%f", e->value);
            break;
        case expr::clock_variable_ee:
            printf("var_%d", static_cast<int>(e->value));
            break;
        case expr::plus_ee:
            printf("(");
            debug_print_expression(e->left, depth + 1);
            printf(" + ");
            debug_print_expression(e->right, depth + 1);
            printf(")");
            break;
        case expr::minus_ee:
            printf("(");
            debug_print_expression(e->left, depth + 1);
            printf(" - ");
            debug_print_expression(e->right, depth + 1);
            printf(")");
            break;
        case expr::multiply_ee:
            printf("(");
            debug_print_expression(e->left, depth + 1);
            printf(" * ");
            debug_print_expression(e->right, depth + 1);
            printf(")");
            break;
        case expr::division_ee:
            printf("(");
            debug_print_expression(e->left, depth + 1);
            printf(" / ");
            debug_print_expression(e->right, depth + 1);
            printf(")");
            break;
        default:
            if(e->operand < expr::random_ee) {
                // It's a value type
                if(e->operand == expr::literal_ee) {
                    printf("%f", e->value);
                } else if(e->operand == expr::clock_variable_ee) {
                    printf("var_%d", static_cast<int>(e->value));
                }
            } else {
                printf("op_%d", e->operand);
            }
    }
}


__global__ void verify_expressions_kernel(SharedModelState* model) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\nVerifying Model Structure with Expressions:\n");
        printf("==========================================\n");

        for(int node_idx = 0; node_idx < model->component_sizes[0]; node_idx++) {
            printf("\nNode Level %d:\n", node_idx);

            for(int comp = 0; comp < model->num_components; comp++) {
                if(node_idx < model->component_sizes[comp]) {
                    const NodeInfo& node = model->nodes[node_idx * model->num_components + comp];
                    if(node.id == -1) continue;

                    printf("\nComponent %d, Node ID %d:\n", comp, node.id);
                    printf("  Type: %d\n", node.type);

                    // Print invariants
                    printf("  Invariants (%d):\n", node.num_invariants);
                    for(int i = 0; i < node.num_invariants; i++) {
                        const GuardInfo& inv = model->invariants[node.first_invariant_index + i];
                        printf("    Invariant %d: ", i);

                        // Print left side (variable)
                        if(inv.uses_variable) {
                            printf("var_%d ", inv.variable_id);
                        }

                        // Print operator
                        switch(inv.operand) {
                            case constraint::less_equal_c: printf("<= "); break;
                            case constraint::less_c: printf("< "); break;
                            case constraint::greater_equal_c: printf(">= "); break;
                            case constraint::greater_c: printf("> "); break;
                            case constraint::equal_c: printf("== "); break;
                            case constraint::not_equal_c: printf("!= "); break;
                            default: printf("op_%d ", inv.operand);
                        }

                        // Print right side expression
                        debug_print_expression(inv.expression);
                        printf("\n");
                    }

                    // Print edges
                    if(node.num_edges > 0) {
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

                            // Print guards
                            printf("      Guards (%d):\n", edge.num_guards);
                            for(int g = 0; g < edge.num_guards; g++) {
                                const GuardInfo& guard = model->guards[edge.guards_start_index + g];
                                printf("        Guard %d: ", g);
                                if(guard.uses_variable) {
                                    printf("var_%d ", guard.variable_id);
                                }

                                switch(guard.operand) {
                                    case constraint::less_equal_c: printf("<= "); break;
                                    case constraint::less_c: printf("< "); break;
                                    case constraint::greater_equal_c: printf(">= "); break;
                                    case constraint::greater_c: printf("> "); break;
                                    case constraint::equal_c: printf("== "); break;
                                    case constraint::not_equal_c: printf("!= "); break;
                                    default: printf("op_%d ", guard.operand);
                                }

                                debug_print_expression(guard.expression);
                                printf("\n");
                            }

                            // Print updates
                            printf("      Updates (%d):\n", edge.num_updates);
                            for(int u = 0; u < edge.num_updates; u++) {
                                const UpdateInfo& update = model->updates[edge.updates_start_index + u];
                                printf("        Update %d: var_%d = ", u, update.variable_id);
                                debug_print_expression(update.expression);
                                printf("\n");
                            }
                        }
                    } else {
                        printf("  No edges\n");
                    }
                }
            }
        }
        printf("\n==========================================\n");
    }
}





__global__ void test_kernel(SharedModelState* model) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Number of components: %d\n", model->num_components);

        // Print nodes in coalesced layout
        for(int node_idx = 0; node_idx < model->component_sizes[0]; node_idx++) {
            printf("\nNode level %d:\n", node_idx);

            for(int comp = 0; comp < model->num_components; comp++) {
                if(node_idx < model->component_sizes[comp]) {
                    const NodeInfo& node = model->nodes[node_idx * model->num_components + comp];

                    // Skip padding nodes
                    if(node.id == -1) continue;

                    // Print node info
                    if(node.num_edges > 0) {
                        printf("Component %d: ID=%d, Type=%d (Edges: %d-%d)\n",
                               comp, node.id, node.type,
                               node.first_edge_index,
                               node.first_edge_index + node.num_edges - 1);
                    } else {
                        printf("Component %d: ID=%d, Type=%d (No edges)\n",
                               comp, node.id, node.type);
                    }


                    // Now use direct edge indexing
                    for(int e = 0; e < node.num_edges; e++) {
                        const EdgeInfo& edge = model->edges[node.first_edge_index + e];
                        printf("  Edge %d: %d -> %d (channel: %d)\n",
                               node.first_edge_index + e, edge.source_node_id,
                               edge.dest_node_id, edge.channel);

                        // Print guards
                        printf("    Guards (%d):\n", edge.num_guards);
                        for(int g = 0; g < edge.num_guards; g++) {
                            const GuardInfo& guard = model->guards[edge.guards_start_index + g];
                            printf("      Guard %d: op=%d, uses_var=%d\n",
                                   g, guard.operand, guard.uses_variable);
                        }

                        // Print updates
                        printf("    Updates (%d):\n", edge.num_updates);
                        for(int u = 0; u < edge.num_updates; u++) {
                            const UpdateInfo& update = model->updates[edge.updates_start_index + u];
                            printf("      Update %d: var_id=%d\n",
                                   u, update.variable_id);
                        }
                    }
                }
            }
        }
    }
}



__global__ void validate_edge_indices(SharedModelState* model) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int total_edges = 0;
        bool indices_valid = true;

        // Validate each node's edge indices
        for(int level = 0; level < model->component_sizes[0]; level++) {
            for(int comp = 0; comp < model->num_components; comp++) {
                if(level < model->component_sizes[comp]) {
                    const NodeInfo& node = model->nodes[level * model->num_components + comp];
                    if(node.id != -1) {
                        // Check edge indices are in range
                        if(node.first_edge_index < 0) {
                            printf("Error: Node %d has negative edge index\n", node.id);
                            indices_valid = false;
                        }

                        // Check edges are sequential
                        if(node.first_edge_index < total_edges) {
                            printf("Error: Node %d edges overlap with previous node\n", node.id);
                            indices_valid = false;
                        }

                        // Verify all edges belong to this node
                        for(int e = 0; e < node.num_edges; e++) {
                            const EdgeInfo& edge = model->edges[node.first_edge_index + e];
                            if(edge.source_node_id != node.id) {
                                printf("Error: Edge %d doesn't belong to node %d\n",
                                       node.first_edge_index + e, node.id);
                                indices_valid = false;
                            }
                        }

                        total_edges += node.num_edges;
                    }
                }
            }
        }

        printf("\n");
        printf("==========================================\n");
        printf("Edge index validation %s\n", indices_valid ? "PASSED" : "FAILED");
        printf("Total edges: %d\n", total_edges);
        printf("==========================================\n");
    }
}


// __global__ void verify_invariants_kernel(SharedModelState* model) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         printf("\nVerifying Invariants:\n");
//         printf("==========================================\n");
//
//         for(int node_idx = 0; node_idx < model->component_sizes[0]; node_idx++) {
//             printf("\nNode Level %d:\n", node_idx);
//
//             for(int comp = 0; comp < model->num_components; comp++) {
//                 if(node_idx < model->component_sizes[comp]) {
//                     const NodeInfo& node = model->nodes[node_idx * model->num_components + comp];
//                     if(node.id == -1) continue;
//
//                     printf("\nComponent %d, Node ID %d:\n", comp, node.id);
//                     printf("  Type: %d\n", node.type);
//                     printf("  Invariants: %d\n", node.num_invariants);
//
//                     for(int i = 0; i < node.num_invariants; i++) {
//                         const GuardInfo& inv = model->invariants[node.first_invariant_index + i];
//                         printf("    Invariant %d:\n", i);
//                         printf("      Guard Info: uses_var=%d, op=%d\n",
//                                inv.uses_variable, inv.operand);
//
//                         // Left side (usually variable)
//                         if(inv.uses_variable) {
//                             printf("      Left side: var_%d\n", inv.variable_id);
//                         }
//
//                         // Operator
//                         printf("      Operator: ");
//                         switch(inv.operand) {
//                             case constraint::less_equal_c: printf("<=\n"); break;
//                             case constraint::less_c: printf("<\n"); break;
//                             case constraint::greater_equal_c: printf(">=\n"); break;
//                             case constraint::greater_c: printf(">\n"); break;
//                             case constraint::equal_c: printf("==\n"); break;
//                             case constraint::not_equal_c: printf("!=\n"); break;
//                             default: printf("op_%d\n", inv.operand);
//                         }
//
//                         // Right side (expression)
//                         printf("      Right side expression ptr: %p\n",
//                                static_cast<void*>(inv.expression));
//                         if(inv.expression != nullptr) {
//                             printf("      Expression details: ");
//                             print_expression_tree(inv.expression);
//                             printf("\n");
//                         }
//                         printf("\n");
//                     }
//                 }
//             }
//         }
//         printf("\n==========================================\n");
//     }
// }







