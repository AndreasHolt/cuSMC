//
// Created by andwh on 09/12/2024.
//

#include "model_verification.cuh"

#include <iostream>

void print_node_info(const node *n, const std::string &prefix = "") {
    std::cout << prefix << "Node ID: " << n->id << " Type: " << n->type << "\n";
    std::cout << prefix << "Edges:\n";
    for (int i = 0; i < n->edges.size; i++) {
        const edge &e = n->edges[i];
        std::cout << prefix << "  -> Dest ID: " << e.dest->id
                << " Channel: " << e.channel << "\n";
    }
}

__device__ void debug_print_expression(const expr *e, int depth = 0) {
    printf("\nDEBUG: Entering debug_print_expression with ptr=%p\n",
           static_cast<const void *>(e));

    if (depth > 10) {
        printf("[max_depth]");
        return;
    }

    if (e == nullptr) {
        printf("[null]");
        return;
    }

    // Try to read the operand value first
    printf("DEBUG: About to read operand at %p\n", static_cast<const void *>(e));
    int op = e->operand;
    printf("DEBUG: Got operand=%d\n", op);

    printf("[");

    // Print operator type - with safety check
    printf("DEBUG: Checking operand...\n");
    if (e->operand >= 0 && e->operand <= expr::pn_skips_ee) {
        switch (e->operand) {
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
           static_cast<const void *>(e->left),
           static_cast<const void *>(e->right));

    // Print subtrees if they exist
    if (e->left != nullptr || e->right != nullptr) {
        printf("DEBUG: Processing children...\n");
        if (e->left != nullptr) {
            printf("left=");
            debug_print_expression(e->left, depth + 1);
        } else {
            printf("left=[null]");
        }

        if (e->right != nullptr) {
            printf(" right=");
            debug_print_expression(e->right, depth + 1);
        } else {
            printf(" right=[null]");
        }
    }

    // Special handling for conditional with extra safety
    if (e->operand == expr::conditional_ee) {
        printf(" else=");
        if (e->conditional_else != nullptr) {
            debug_print_expression(e->conditional_else, depth + 1);
        } else {
            printf("[null]");
        }
    }

    printf("]");
}

__global__ void verify_expressions_kernel(SharedModelState *model) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\nVerifying Model Structure with Variable Types:\n");
        printf("==========================================\n");

        printf("Number of components: %d\n", model->num_components);
        for (int comp = 0; comp < model->num_components; comp++) {
            printf("Component %d size: %d\n", comp, model->component_sizes[comp]);
        }

        for (int node_idx = 0; node_idx < model->component_sizes[0]; node_idx++) {
            printf("\nProcessing Node Level %d:\n", node_idx);

            for (int comp = 0; comp < model->num_components; comp++) {
                printf("  Checking component %d...\n", comp);

                if (node_idx < model->component_sizes[comp]) {
                    const NodeInfo &node = model->nodes[node_idx * model->num_components + comp];
                    if (node.id == -1) {
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
                    for (int i = 0; i < node.num_invariants; i++) {
                        printf("    Processing invariant %d...\n", i);

                        if (node.first_invariant_index + i >= 0) {
                            printf("DEBUG: Accessing invariant at index %d\n", i);
                            const GuardInfo &inv = model->invariants[node.first_invariant_index + i];
                            printf("DEBUG: Got invariant, uses_variable=%d, operand=%d\n",
                                   inv.uses_variable, inv.operand);
                            printf("DEBUG: About to print expression at %p\n",
                                   static_cast<const void *>(inv.expression));
                            printf("    Invariant %d: ", i);

                            if (inv.uses_variable) {
                                printf("%s [%s] (ID: %d) ",
                                       inv.var_info.name,
                                       inv.var_info.type == VariableKind::CLOCK ? "CLOCK" : "INT",
                                       inv.var_info.variable_id);
                            }

                            printf("operator: %d, expression: ", inv.operand);
                            debug_print_expression(inv.expression);
                            printf("\n");
                        } else {
                            printf("    Invalid invariant index!\n");
                        }
                    }

                    // Print edges with safety checks
                    printf("  Processing edges...\n");
                    if (node.num_edges > 0 && node.first_edge_index >= 0) {
                        printf("  Edges (%d-%d):\n",
                               node.first_edge_index,
                               node.first_edge_index + node.num_edges - 1);

                        for (int e = 0; e < node.num_edges; e++) {
                            const EdgeInfo &edge = model->edges[node.first_edge_index + e];
                            printf("    Edge %d: %d -> %d (channel: %d)\n",
                                   node.first_edge_index + e,
                                   edge.source_node_id,
                                   edge.dest_node_id,
                                   edge.channel);

                            // Print guards with safety checks
                            if (edge.num_guards > 0 && edge.guards_start_index >= 0) {
                                printf("      Guards (%d):\n", edge.num_guards);
                                for (int g = 0; g < edge.num_guards; g++) {
                                    const GuardInfo &guard = model->guards[edge.guards_start_index + g];
                                    printf("        Guard %d: ", g);

                                    if (guard.uses_variable) {
                                        printf("%s [%s] (ID: %d) ",
                                               guard.var_info.name,
                                               guard.var_info.type == VariableKind::CLOCK ? "CLOCK" : "INT",
                                               guard.var_info.variable_id);
                                    }

                                    printf("operator: %d, ", guard.operand);
                                    if (guard.expression) {
                                        printf("expression: ");
                                        debug_print_expression(guard.expression);
                                    } else {
                                        printf("(null expression)");
                                    }
                                    printf("\n");
                                }
                            }

                            // Print updates with safety checks
                            if (edge.num_updates > 0 && edge.updates_start_index >= 0) {
                                printf("      Updates (%d):\n", edge.num_updates);
                                for (int u = 0; u < edge.num_updates; u++) {
                                    const UpdateInfo &update = model->updates[edge.updates_start_index + u];
                                    printf("        Update %d: var_%d [%s] = ",
                                           u,
                                           update.variable_id,
                                           update.kind == VariableKind::CLOCK ? "CLOCK" : "INT");

                                    if (update.expression) {
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