//
// Created by andwh on 01/11/2024.
//

#ifndef DATA_STRUCTURES_CUH
#define DATA_STRUCTURES_CUH
#include <curand_kernel.h>
#include <stdint.h>


class data_structures {
};

// Constants for configuration
constexpr int MAX_COMPONENTS = 32; // One warp
constexpr int MAX_CLOCKS = 8; // Clocks per component
constexpr int MAX_GLOBAL_VARS = 32; // Shared variables
constexpr int MAX_CHANNELS = 32; // Communication channels
constexpr int WARP_SIZE = 32;

__shared__ struct {
    // 1. Component States (SOA layout)
    struct {
        // Core state data
        uint16_t locations[MAX_COMPONENTS]; // Current location per component
        uint8_t flags[MAX_COMPONENTS]; // Component status flags
        int16_t clocks[MAX_COMPONENTS][MAX_CLOCKS]; // Component-specific clocks

        // Packed status flags for each component
        union {
            struct {
                uint32_t is_urgent: 1;
                uint32_t is_committed: 1;
                uint32_t needs_sync: 1;
                uint32_t is_active: 1;
                uint32_t reserved: 28;
            } bits[MAX_COMPONENTS];

            uint32_t packed[MAX_COMPONENTS];
        } status;
    } states;

    // 2. Global Variables
    // Shared across components, since semantically the set of variables are not disjoint across components
    struct {
        int16_t values[MAX_GLOBAL_VARS]; // Actual variable values
        uint32_t modified_mask; // Track which variables were modified
    } variables;

    // 3. Timing & Delay Management
    struct {
        // Current delays
        union {
            struct {
                float delays[MAX_COMPONENTS]; // Sampled delays
                int component_ids[MAX_COMPONENTS]; // Component mapping
                uint32_t valid_mask; // Which delays are valid
            };

            struct {
                float delay;
                int component_id;
                bool is_valid;
            } delay_info[MAX_COMPONENTS]; // Alternative view
        } current;

        // Winner tracking
        struct {
            float min_delay; // Winning delay
            int winner_id; // Winning component
            bool winner_found; // Valid winner exists
        } winner;

        float global_time; // Current simulation time
    } timing;

    // 4. Synchronization Management
    struct {
        // Request double buffer
        struct {
            // Active synchronization requests
            struct {
                union {
                    struct {
                        uint32_t channel: 10; // Channel ID
                        uint32_t component: 10; // Component ID
                        uint32_t is_sender: 1; // ! or ?
                        uint32_t is_urgent: 1; // Urgent channel
                        uint32_t is_broadcast: 1; // Broadcast channel
                        uint32_t is_active: 1; // Request is valid
                        uint32_t reserved: 8;
                    } bits;

                    uint32_t packed;
                } info;

                float delay; // Associated delay
            } requests[2][MAX_COMPONENTS]; // Double buffered

            uint32_t count[2]; // Number of requests in each buffer
            uint32_t active_buffer; // Which buffer is active
        } sync_requests;

        // Broadcast state
        struct {
            uint32_t receivers[MAX_COMPONENTS / 32]; // Bit vector of receivers
            uint16_t sender; // Broadcasting component
            uint16_t channel; // Broadcast channel
            uint32_t num_receivers; // Receiver count
            bool active; // Broadcast in progress
        } broadcast;

        // Binary synchronization tracking
        struct {
            uint16_t sender;
            uint16_t receiver;
            uint16_t channel;
        } matches[MAX_COMPONENTS / 2]; // Potential matches
        uint32_t match_count; // Number of valid matches
    } sync;

    // 5. Progress & Control
    struct {
        uint32_t step_count; // Simulation steps taken
        uint32_t active_components; // Number of active components
        uint32_t enabled_mask; // Bit mask of enabled components
        bool simulation_complete; // Termination flag
    } control;

    // 6. Random State (one per warp for efficiency)
    curandState rand_states[MAX_COMPONENTS / WARP_SIZE];
} shared_data;

#endif //DATA_STRUCTURES_CUH
