//
// Created by andwh on 04/11/2024.
//

#ifndef MAIN_CUH
#define MAIN_CUH

#include <iostream>



#define VERBOSE 0
#define EXPR_VERBOSE 0 // this
#define REDUCTION_VERBOSE 0
#define EXPR_VERBOSE_EXTRA 0
#define DELAY_VERBOSE 0
#define CHANNEL_VERBOSE 0
#define PRINT_TRANSITIONS 0
#define PRINT_TRANS_ALOHA 0 // this
#define PRINT_VARIABLES 0
#define PRINT_UPDATES 0 // this
#define LISTEN_TO -1 //-1 = all
#define QUERYSTATS 0
#define MINIMAL_PRINTS 0
#define TIME_BOUND 10.0
#define MAX_VARIABLES 3
#define SYNC_SIDE_EFFECT true

constexpr int MAX_EDGES_PER_NODE = 3;
constexpr int MAX_CHANNELS = 2;
constexpr bool USE_GLOBAL_MEMORY_CURAND = true;

struct configuration {
    std::string filename;
    int curand_seed;
};

struct model_info {
    int MAX_VALUE_STACK_SIZE = 64;
    //int MAX_CHANNELS = 5;
    //int MAX_EDGES_PER_NODE = 8;
    int runs_per_block = 1;
    uint32_t num_vars;
};

struct statistics_Configuration {
    // Statistics
    int simulations;
    int timeBound;
    int variable_threshhold;
    int variable_id;
    bool isMax; // Gather info on either the max value of the variable or the min
    bool isEstimate;
    std::string loc_query;
    statistics_Configuration() : simulations(0), timeBound(0), variable_threshhold(0), variable_id(0), isMax(false), isEstimate(false), loc_query("") {}
    statistics_Configuration(int sims, int tBound, int vThresh, int vId, bool max, bool estimate, const std::string& locQuery)
        : simulations(sims), timeBound(tBound), variable_threshhold(vThresh), variable_id(vId), isMax(max), isEstimate(estimate), loc_query(locQuery) {}
};

bool HandleCommandLineArguments(int argc, char **argv, std::string *filename, int *seed, statistics_Configuration* stats, bool CONST_QUERY);

#endif //MAIN_CUH
