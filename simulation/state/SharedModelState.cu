//
// Created by andwh on 04/11/2024.
//

#include "SharedModelState.cuh"


SharedModelState* init_shared_model_state(
    const network* cpu_network,
    const std::unordered_map<int, int>& node_subsystems_map,
    const std::unordered_map<int, node*>& node_map)
{
    // First organize nodes by component
    std::vector<std::vector<node*>> components_nodes;
    int max_component_id = -1;

    // Find number of components
    for(const auto& pair : node_subsystems_map) {
        max_component_id = std::max(max_component_id, pair.second);
    }
    components_nodes.resize(max_component_id + 1);

    // Group nodes by component
    for(const auto& pair : node_map) {
        int node_id = pair.first;
        node* node_ptr = pair.second;
        int component_id = node_subsystems_map.at(node_id);
        components_nodes[component_id].push_back(node_ptr);
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

    // Allocate device memory for NodeInfo array
    // Organized as: [comp0_node0, comp1_node0, comp2_node0, comp0_node1, ...]
    const int total_node_slots = max_nodes_per_component * components_nodes.size();
    NodeInfo* device_nodes;
    cudaMalloc(&device_nodes, total_node_slots * sizeof(NodeInfo));

    // Create and copy NodeInfo for each node in coalesced layout
    std::vector<NodeInfo> host_nodes(total_node_slots);
    for(int node_idx = 0; node_idx < max_nodes_per_component; node_idx++) {
        for(int comp_idx = 0; comp_idx < components_nodes.size(); comp_idx++) {
            int array_idx = node_idx * components_nodes.size() + comp_idx;

            if(node_idx < components_nodes[comp_idx].size()) {
                node* cpu_node = components_nodes[comp_idx][node_idx];
                host_nodes[array_idx] = NodeInfo(
                    cpu_node->id,
                    cpu_node->type,
                    cpu_node->lamda
                );

            } else {
                // Padding for components with fewer nodes
                host_nodes[array_idx] = NodeInfo{-1, -1, nullptr};
            }
        }
    }

    // Copy NodeInfo array to device
    cudaMemcpy(device_nodes, host_nodes.data(),
               total_node_slots * sizeof(NodeInfo),
               cudaMemcpyHostToDevice);

    // Create and copy SharedModelState
    SharedModelState host_model{
        static_cast<int>(components_nodes.size()),     // num_components
        device_component_sizes,      // component_sizes
        device_nodes                 // nodes
        // later add edge info
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
                    printf("Component %d: ID=%d, Type=%d\n",
                           comp, node.id, node.type);
                }
            }
        }
    }
}

