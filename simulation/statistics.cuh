//
// Created by andwh on 09/12/2024.
//

#ifndef STATISTICS_CUH
#define STATISTICS_CUH
#include <iostream>
#include <stdexcept>
#include "../main.cuh"

enum StatType {
    NO_STAT,
    COMP_STAT,
    VAR_STAT
};

struct var_at_time {
    float value;
    float time;
};

class Statistics {
private:
    size_t simulations;
    StatType stat_type;

    // For component queries
    bool *goal_flags_host_ptr;
    bool *goal_flags_device_ptr;

    // For variable queries
    double *variable_value_host_ptr;
    double *variable_value_device_ptr;

    void cleanup() {
        if (goal_flags_host_ptr) free(goal_flags_host_ptr);
        if (goal_flags_device_ptr) cudaFree(goal_flags_device_ptr);
    }

    void cleanup_var() {
        free(variable_value_host_ptr);
        cudaFree(variable_value_device_ptr);
    }

public:
    // Allocate memory for goal flags in constructor
    Statistics(size_t num_simulations, StatType stat_type) : simulations(num_simulations), stat_type(stat_type) {
        if (stat_type == COMP_STAT) {
            // Initialize arrays
            goal_flags_host_ptr = (bool *) malloc(simulations * sizeof(bool));
            if (!goal_flags_host_ptr) {
                throw std::runtime_error("(STATS: COMP) Host memory allocation failed");
            }
            memset(goal_flags_host_ptr, 0, simulations * sizeof(bool)); // memset just initializes the values to 0

            cudaError_t error = cudaMalloc(&goal_flags_device_ptr, simulations * sizeof(bool));
            if (error != cudaSuccess) {
                free(goal_flags_host_ptr);
                throw std::runtime_error("(STATS: COMP) Device memory allocation failed: " +
                                         std::string(cudaGetErrorString(error)));
            }

            error = cudaMemcpy(goal_flags_device_ptr, goal_flags_host_ptr,
                               simulations * sizeof(bool), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                cleanup();
                throw std::runtime_error("(STATS: COMP) Initial memory copy failed: " +
                                         std::string(cudaGetErrorString(error)));
            }
        } else if (stat_type == VAR_STAT) {
            // Initialize arrays
            variable_value_host_ptr = (double *) malloc(simulations * sizeof(double));
            if (!variable_value_host_ptr) {
                throw std::runtime_error("(STATS: VARIABLE) Host memory allocation failed");
            }

            for(size_t i = 0; i < simulations; i++) {
                variable_value_host_ptr[i] = 0.0;
            }

            cudaError_t error = cudaMalloc(&variable_value_device_ptr, simulations * sizeof(double));
            if (error != cudaSuccess) {
                free(variable_value_host_ptr);
                throw std::runtime_error("(STATS: VARIABLE) Device memory allocation failed: " +
                                         std::string(cudaGetErrorString(error)));
            }

            error = cudaMemcpy(variable_value_device_ptr, variable_value_host_ptr,
                               simulations * sizeof(double), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                cleanup_var();
                throw std::runtime_error("(STATS: VARIABLE) Initial memory copy failed: " +
                                         std::string(cudaGetErrorString(error)));
            }

        }
    }

    ~Statistics() {
        if (stat_type == COMP_STAT) {
            cleanup();
        }
        if (stat_type == VAR_STAT) {
            cleanup_var();
        }
    }

    bool *get_comp_device_ptr() {
        if (stat_type == COMP_STAT) {
            return goal_flags_device_ptr;
        }

        return nullptr;
    }

    double *get_var_device_ptr() {

        if (stat_type == VAR_STAT) {
            return variable_value_device_ptr;
        }

        return nullptr;
    }

    struct Results {
        int hits;
        float probability;
        size_t total_simulations;
    };

    Results collect_results() {
        if (stat_type == COMP_STAT) {
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

        throw std::runtime_error("Stat type not supported");

    }

    double *collect_var_data() {
        cudaError_t error = cudaMemcpy(variable_value_host_ptr, variable_value_device_ptr,
                                           simulations * sizeof(double), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            throw std::runtime_error("(STATS: VARIABLE) Memory copy from device failed: " +
                                     std::string(cudaGetErrorString(error)));
        }

        return variable_value_host_ptr;
    }

    void print_results(const std::string &query, const Results &results) {
        std::cout << "Number of simulations: " << results.total_simulations << std::endl;
        std::cout << "Number of hits: " << results.hits << std::endl;
        std::cout << "The answer to " << query << " is " << results.probability << std::endl;
    }
};


#endif //STATISTICS_CUH
