//
// Created by andwh on 04/11/2024.
//

#ifndef MAIN_CUH
#define MAIN_CUH

#define VERBOSE 0
#define MINIMAL_PRINTS 0



struct conf {
    std::string filename;
    bool verbose;
    bool debug;
    int currand_seed;
};
extern const conf configuration;


#endif //MAIN_CUH
