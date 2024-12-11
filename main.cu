//
// Created by andwh on 04/11/2024.
//

#include "main.cuh"
#include <chrono>
#include "include/engine/domain.h"
#include "automata_parser/uppaal_xml_parser.h"
#include <iostream>
#include "automata_parser/network/network_props.h"
#include "automata_parser/network/domain_optimization_visitor.h"
#include "automata_parser/network/pn_compile_visitor.h"
#include "simulation/simulation.cuh"
#include "simulation/state/shared_model_state.cuh"
#include "automata_parser/variable_usage_visitor.h"
#include "simulation/statistics.cuh"
#include "smc.cuh"

int main(int argc, char *argv[]) {
    std::string filename = "../xml_files/UppaalBehaviorTest3.xml";
    int currand_seed = 0;
    //int MAX_COMPONENTS = 3;
    // Statistics
    int simulations = 1000;
    bool isMax = true; // Gather info on either the max value of the variable or the min

    bool succeded = HandleCommandLineArguments(argc, argv, &filename, &currand_seed, &simulations, &isMax);
    if (!succeded) {return 1;}

    const struct configuration conf = {filename, currand_seed, simulations, isMax};

    //int simulation_arr[] = {1, 10};

    //for (int i : simulation_arr) {
    smc(filename, "", true, false, 0, 5, conf);
    //}


    return 1;
}


bool HandleCommandLineArguments(int argc, char **argv, string *filename, int *seed, int * runs, bool *isMax) {
    for (int i = 1; i < argc; i++) {    // Skip first argument, which is the executable path.
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                *filename = argv[i + 1];
                i++; // Skip the next argument
            } else {
                std::cerr << "Error: -i option requires a value" << std::endl;
                return false;
            }
        } else if (arg == "-s" || arg == "--seed") {
            if (i + 1 < argc) {
                std::string str = argv[i + 1];
                i++; // Skip the next argument
                try {
                    const int num = std::stoi(str);
                    *seed = num;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: invalid argument: " << e.what() << std::endl;
                    return false;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Error: number too big: " << e.what() << std::endl;
                    return false;
                }

            } else {
                std::cerr << "Error: -s option requires a value" << std::endl;
                return false;
            }
        } else if (arg == "-r" || arg == "--runs") {
            if (i + 1 < argc) {
                std::string str = argv[i + 1];
                i++; // Skip the next argument
                try {
                    int num = std::stoi(str);
                    *runs = num;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: invalid argument: " << e.what() << std::endl;
                    return false;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Error: number too big: " << e.what() << std::endl;
                    return false;
                }

            } else {
                std::cerr << "Error: -r option requires a value" << std::endl;
                return false;
            }
        } else if (arg == "--min") {
            *isMax = false;
        } else if (arg == "--max") {
            if (*isMax == false) {
                std::cerr << "Can not use --max and --min at the same time." << std::endl;
                return false;
            }
            *isMax = true;
        } else if (arg == "-h" || arg == "--help") {
            cout << "Use -m or --model, followed by a path, for inputting a path the the model xml file." << endl;
            cout << "Use -s or --seed, followed by a number, to initialize currand with a constant seed. (0 = random seed)" << endl;
            cout << "Use -r or --runs, to specify the number of simulations." << endl;
            cout << "Use --max or --min, to specify whether we want to query for the max value a variable reaches or the lowest." << endl;
            return false;
        }
        else {
            std::cerr << "Error: unknown option " << arg << std::endl;
            return false;
        }
    }
    return true;
}
