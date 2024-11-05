//
// Created by andwh on 04/11/2024.
//

#include "SharedModelState.cuh"

#include <iostream>

#define MAX_NODES_PER_COMPONENT 5

void count_edges_and_constraints(
    const std::vector<std::vector<node*>>& components_nodes,
    int& total_edges,
    int& total_guards,
    int& total_updates,
    std::vector<int>& edges_per_node,
    std::vector<int>& node_edge_starts)
{
    total_edges = 0;
    total_guards = 0;
    total_updates = 0;

    // For each node level
    for(int node_idx = 0; node_idx < MAX_NODES_PER_COMPONENT; node_idx++) {
        // For each component
        for(int comp_idx = 0; comp_idx < components_nodes.size(); comp_idx++) {
            if(node_idx < components_nodes[comp_idx].size()) {
                node* current_node = components_nodes[comp_idx][node_idx];

                // Store start index for this node's edges
                node_edge_starts.push_back(total_edges);
                edges_per_node.push_back(current_node->edges.size);

                // Count guards and updates for each edge
                for(int e = 0; e < current_node->edges.size; e++) {
                    total_edges++;
                    total_guards += current_node->edges[e].guards.size;
                    total_updates += current_node->edges[e].updates.size;
                }
            }
        }
    }
}

void fill_edge_arrays(
    const std::vector<std::vector<node*>>& components_nodes,
    std::vector<EdgeInfo>& host_edges,
    std::vector<GuardInfo>& host_guards,
    std::vector<UpdateInfo>& host_updates)
{
    int current_guard_index = 0;
    int current_update_index = 0;

    // For each node level
    for(int node_idx = 0; node_idx < MAX_NODES_PER_COMPONENT; node_idx++) {
        // For each component
        for(int comp_idx = 0; comp_idx < components_nodes.size(); comp_idx++) {
            if(node_idx < components_nodes[comp_idx].size()) {
                node* current_node = components_nodes[comp_idx][node_idx];

                // Add edges for this node in coalesced layout
                for(int e = 0; e < current_node->edges.size; e++) {
                    const edge& current_edge = current_node->edges[e];

                    // Store edge info
                    host_edges.emplace_back(
                        current_node->id,              // source_node_id
                        current_edge.dest->id,         // dest_node_id
                        current_edge.channel,          // channel
                        current_edge.weight,           // weight
                        current_edge.guards.size,      // num_guards
                        current_guard_index,           // guards_start_index
                        current_edge.updates.size,     // num_updates
                        current_update_index           // updates_start_index
                    );

                    // Store guards
                    for(int g = 0; g < current_edge.guards.size; g++) {
                        const constraint& guard = current_edge.guards[g];
                        host_guards.push_back(GuardInfo{
                            guard.operand,
                            guard.uses_variable,
                            guard.value,
                            guard.expression
                        });
                        current_guard_index++;
                    }

                    // Store updates
                    for(int u = 0; u < current_edge.updates.size; u++) {
                        const update& upd = current_edge.updates[u];
                        host_updates.push_back(UpdateInfo{
                            upd.variable_id,
                            upd.expression
                        });
                        current_update_index++;
                    }
                }
            }
        }
    }
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



SharedModelState* init_shared_model_state(
    const network* cpu_network,
    const std::unordered_map<int, int>& node_subsystems_map,
    const std::unordered_map<int, std::list<edge>>& node_edge_map)
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

    // Count total edges and allocate edge info
    // int total_edges = 0;
    // for(const auto& pair : node_edge_map) {
    //     total_edges += pair.second.size();
    // }

    // // Allocate device memory for NodeInfo and EdgeInfo arrays
    const int total_node_slots = max_nodes_per_component * components_nodes.size();
    // NodeInfo* device_nodes;
    // EdgeInfo* device_edges;
    // cudaMalloc(&device_nodes, total_node_slots * sizeof(NodeInfo));
    // cudaMalloc(&device_edges, total_edges * sizeof(EdgeInfo));
    //
    // // Create and copy NodeInfo and EdgeInfo in coalesced layout
    // std::vector<NodeInfo> host_nodes;
    // std::vector<EdgeInfo> host_edges;
    // host_nodes.reserve(total_node_slots);
    // host_edges.reserve(total_edges);
    //
    int current_edge_index = 0;

    // Count total guards and updates
    int total_edges = 0;
    int total_guards = 0;
    int total_updates = 0;
    for(const auto& pair : node_edge_map) {
        for(const auto& edge : pair.second) {
            total_edges++;
            total_guards += edge.guards.size;
            total_updates += edge.updates.size;
        }
    }

    // Allocate device memory
    NodeInfo* device_nodes;
    EdgeInfo* device_edges;
    GuardInfo* device_guards;
    UpdateInfo* device_updates;
    cudaMalloc(&device_nodes, total_node_slots * sizeof(NodeInfo));
    cudaMalloc(&device_edges, total_edges * sizeof(EdgeInfo));
    cudaMalloc(&device_guards, total_guards * sizeof(GuardInfo));
    cudaMalloc(&device_updates, total_updates * sizeof(UpdateInfo));

    // Create host arrays
    std::vector<NodeInfo> host_nodes;
    std::vector<EdgeInfo> host_edges;
    std::vector<GuardInfo> host_guards;
    std::vector<UpdateInfo> host_updates;
    host_nodes.reserve(total_node_slots);
    host_edges.reserve(total_edges);
    host_guards.reserve(total_guards);
    host_updates.reserve(total_updates);

    int current_guard_index = 0;
    int current_update_index = 0;

    // for each node level
    for(int node_idx = 0; node_idx < max_nodes_per_component; node_idx++) {
        // for each component at this node level
        for(int comp_idx = 0; comp_idx < components_nodes.size(); comp_idx++) {
            if(node_idx < components_nodes[comp_idx].size()) {
                const auto& node_pair = components_nodes[comp_idx][node_idx];
                int node_id = node_pair.first;
                const std::list<edge>& edges = *node_pair.second;

                // Get node info from first edge's dest if available
                node::node_types node_type = node::location; // default
                expr* node_lambda = nullptr;
                if(!edges.empty()) {
                    const node* dest_node = edges.front().dest;
                    if(dest_node->id == node_id) {
                        node_type = dest_node->type;
                        node_lambda = dest_node->lamda;
                    }
                }

                // Create NodeInfo
                NodeInfo node_info{node_id, node_type, node_lambda};
                host_nodes.push_back(node_info);

                // Add edges and their guards/updates
                for(const edge& e : edges) {
                    // Store guards
                    int guards_start = current_guard_index;
                    for(int g = 0; g < e.guards.size; g++) {
                        const constraint& guard = e.guards[g];
                        host_guards.push_back(GuardInfo{
                            guard.operand,
                            guard.uses_variable,
                            guard.value,
                            guard.expression
                        });
                        current_guard_index++;
                    }

                    // Store updates
                    int updates_start = current_update_index;
                    for(int u = 0; u < e.updates.size; u++) {
                        const update& upd = e.updates[u];
                        host_updates.push_back(UpdateInfo{
                            upd.variable_id,
                            upd.expression
                        });
                        current_update_index++;
                    }

                    // Create edge info
                    EdgeInfo edge_info{
                        node_id,
                        e.dest->id,
                        e.channel,
                        e.weight,
                        e.guards.size,
                        guards_start,
                        e.updates.size,
                        updates_start
                    };
                    host_edges.push_back(edge_info);
                }
            } else {
                // Padding for components with fewer nodes
                host_nodes.push_back(NodeInfo{-1, node::location, nullptr});
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

    // Create and copy SharedModelState
    SharedModelState host_model{
        static_cast<int>(components_nodes.size()),
        device_component_sizes,
        device_nodes,
        device_edges,
        nullptr,  // todo: remove edges_per_node no longer needed
        nullptr,  // todo: remove node_edge_starts no longer needed
        device_guards,
        device_updates
    };

    SharedModelState* device_model;
    cudaMalloc(&device_model, sizeof(SharedModelState));
    cudaMemcpy(device_model, &host_model, sizeof(SharedModelState),
               cudaMemcpyHostToDevice);

    return device_model;
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
                    printf("Component %d: ID=%d, Type=%d\n",
                           comp, node.id, node.type);

                    // Find edges for this node
                    int edge_idx = 0;
                    const EdgeInfo* edge;
                    while((edge = &model->edges[edge_idx]) != nullptr) {
                        // More strict validation of edge data
                        if(edge->source_node_id <= 0 || edge->dest_node_id <= 0 ||
                           edge->channel < 0 || edge->channel > 100 ||  // Assuming reasonable channel range
                           edge->num_guards < 0 || edge->num_guards > 10 ||  // Assuming reasonable guard count
                           edge->num_updates < 0 || edge->num_updates > 10) {  // Assuming reasonable update count
                            break;
                        }

                        if(edge->source_node_id == node.id) {
                            printf("  Edge %d: %d -> %d (channel: %d)\n",
                                   edge_idx, edge->source_node_id,
                                   edge->dest_node_id, edge->channel);

                            // Print guards
                            printf("    Guards (%d):\n", edge->num_guards);
                            for(int g = 0; g < edge->num_guards; g++) {
                                const GuardInfo& guard = model->guards[edge->guards_start_index + g];
                                printf("      Guard %d: op=%d, uses_var=%d\n",
                                       g, guard.operand, guard.uses_variable);
                            }

                            // Print updates
                            printf("    Updates (%d):\n", edge->num_updates);
                            for(int u = 0; u < edge->num_updates; u++) {
                                const UpdateInfo& update = model->updates[edge->updates_start_index + u];
                                printf("      Update %d: var_id=%d\n",
                                       u, update.variable_id);
                            }
                        }
                        edge_idx++;

                        // Additional safety check
                        if(edge_idx > 100) break;  // Assuming no more than 100 edges total
                    }
                }
            }
        }
    }
}







