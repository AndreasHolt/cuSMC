//
// Created by andwh on 04/11/2024.
//

#ifndef MAIN_CUH
#define MAIN_CUH

#include <iostream>



#define VERBOSE 0
#define MINIMAL_PRINTS 0
#define TIME_BOUND 10.0
#define MAX_VARIABLES 20

constexpr int MAX_EDGES_PER_NODE = 8;
constexpr int MAX_CHANNELS = 5;
constexpr bool USE_GLOBAL_MEMORY_CURAND = true;

struct configuration {
    std::string filename;
    int curand_seed;
    int simulations;
    bool isMax;
};
extern const configuration conf;

struct model_info {
    int MAX_COMPONENTS = 3;
    int MAX_VALUE_STACK_SIZE = 64;
    //int MAX_CHANNELS = 5;
    //int MAX_EDGES_PER_NODE = 8;
    int runs_per_block = 1;
};
extern const model_info m_info;


bool HandleCommandLineArguments(int argc, char **argv, std::string *filename, int *seed, int * runs, bool *isMax);

#endif //MAIN_CUH
