//
// Created by andwh on 04/11/2024.
//

#include "SharedModelState.cuh"


SharedModelState* init_shared_model_state(
    const network* cpu_network,
    const std::unordered_map<int, int>& node_subsystems_map,
    const std::unordered_map<int, node*>& node_map) {

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

    // Allocate device memory
    const node** device_nodes;
    int* device_starts;
    int* device_sizes;
    cudaMalloc(&device_nodes, node_map.size() * sizeof(node*));
    cudaMalloc(&device_starts, components_nodes.size() * sizeof(int));
    cudaMalloc(&device_sizes, components_nodes.size() * sizeof(int));

    // Host arrays for component info
    std::vector<int> component_starts(components_nodes.size());
    std::vector<int> component_sizes(components_nodes.size());

    // Copy nodes by component
    int current_index = 0;
    for(int i = 0; i < components_nodes.size(); i++) {
        component_starts[i] = current_index;
        component_sizes[i] = components_nodes[i].size();

        for(node* n : components_nodes[i]) {
            // Allocate and copy node
            node* device_node;
            cudaMalloc(&device_node, sizeof(node));
            cudaMemcpy(device_node, n, sizeof(node), cudaMemcpyHostToDevice);

            // Store pointer in nodes array
            cudaMemcpy(&device_nodes[current_index], &device_node,
                      sizeof(node*), cudaMemcpyHostToDevice);
            current_index++;
        }
    }

    // Copy component info to device
    cudaMemcpy(device_starts, component_starts.data(),
               component_starts.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_sizes, component_sizes.data(),
               component_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Create and copy SharedModelState
    SharedModelState host_model(device_nodes, node_map.size(),
                              components_nodes.size(),
                              device_starts, device_sizes);

    SharedModelState* device_model;
    cudaMalloc(&device_model, sizeof(SharedModelState));
    cudaMemcpy(device_model, &host_model, sizeof(SharedModelState),
               cudaMemcpyHostToDevice);

    return device_model;
}


__global__ void test_kernel(SharedModelState* model) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Total nodes: %d\n", model->total_nodes);
        printf("Number of components: %d\n", model->num_components);

        for(int c = 0; c < model->num_components; c++) {
            printf("\nComponent %d:\n", c);
            int start = model->component_starts[c];
            int size = model->component_sizes[c];

            for(int i = 0; i < size; i++) {
                printf("Node %d ID: %d Type: %d\n", i, model->nodes[start + i]->id, model->nodes[start + i]->type);
            }
        }
    }
}