//
// Created by andwh on 09/12/2024.
//

#ifndef STATISTICS_CUH
#define STATISTICS_CUH
#include <iostream>
#include <stdexcept>
#include "../main.cuh"

class Statistics {
private:
    size_t simulations;
    bool* goal_flags_host_ptr;
    bool* goal_flags_device_ptr;

    void cleanup() {
        if (goal_flags_host_ptr) free(goal_flags_host_ptr);
        if (goal_flags_device_ptr) cudaFree(goal_flags_device_ptr);
    }

public:
    // Allocate memory for goal flags in constructor
    Statistics(size_t num_simulations) : simulations(num_simulations) {
        // Initialize arrays
        goal_flags_host_ptr = (bool*)malloc(simulations * sizeof(bool));
        if (!goal_flags_host_ptr) {
            throw std::runtime_error("Host memory allocation failed");
        }
        memset(goal_flags_host_ptr, 0, simulations * sizeof(bool)); // memset just initializes the values to 0

        cudaError_t error = cudaMalloc(&goal_flags_device_ptr, simulations * sizeof(bool));
        if (error != cudaSuccess) {
            free(goal_flags_host_ptr);
            throw std::runtime_error("(STATS) Device memory allocation failed: " +
                                   std::string(cudaGetErrorString(error)));
        }

        error = cudaMemcpy(goal_flags_device_ptr, goal_flags_host_ptr,
                          simulations * sizeof(bool), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            cleanup();
            throw std::runtime_error("(STATS) Initial memory copy failed: " +
                                   std::string(cudaGetErrorString(error)));
        }
    }

    ~Statistics() {
        cleanup();
    }

    bool* get_device_ptr() { return goal_flags_device_ptr; }

    struct Results {
        int hits;
        float probability;
        size_t total_simulations;
    };

    Results collect_results() {
        cudaError_t error = cudaMemcpy(goal_flags_host_ptr, goal_flags_device_ptr,
                                      simulations * sizeof(bool), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            throw std::runtime_error("(STATS) Memory copy from device failed: " +
                                   std::string(cudaGetErrorString(error)));
        }

        int hits = 0;
        for (size_t i = 0; i < simulations; i++) {
            if constexpr (VERBOSE) {
                printf("Flag in host is %d\n", goal_flags_host_ptr[i]);
            }
            if (goal_flags_host_ptr[i]) hits++;
        }

        return Results{
            hits,
            static_cast<float>(hits) / static_cast<float>(simulations),
            simulations
        };
    }

    void print_results(const std::string& query, const Results& results) {
        std::cout << "Number of simulations: " << results.total_simulations << std::endl;
        std::cout << "Number of hits: " << results.hits << std::endl;
        std::cout << "The answer to " << query << " is " << results.probability << std::endl;
    }
};






#endif //STATISTICS_CUH
