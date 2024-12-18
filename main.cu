//
// Created by andwh on 04/11/2024.
//

#include "main.cuh"
#include "smc.cuh"
#include "automata_parser/uppaal_xml_parser.h"
#include <regex>

int main(int argc, char *argv[]) {

    bool CONST_QUERY = true;
    std::string filename = "../xml_files/LargeModels/abc_100_1.xml";
    int curand_seed = 1234;

    // Statistics
    int simulations = 1000;
    int timeBound = 100;
    int variable_threshhold = -1;
    int variable_id = -1;
    bool isMax = true; // Gather info on either the max value of the variable or the min
    bool isEstimate = true;
    string loc_query = "Process0.SUCCESS";

    statistics_Configuration* stat_input = {};

    bool cliSucceeded = HandleCommandLineArguments(argc, argv, &filename, &curand_seed, stat_input, &CONST_QUERY);
    if (!cliSucceeded) {return 0;}

    const struct configuration conf = {filename, curand_seed};
    int NumberOfQueries = 1;
    struct statistics_Configuration* stats = nullptr;
    if (CONST_QUERY) {
        stats = new statistics_Configuration[NumberOfQueries];
        stats[0] = statistics_Configuration(simulations, timeBound, variable_threshhold, variable_id, isMax, isEstimate, loc_query);
        // stats[1] = {100, 100, 12, -1, false, false, "c1.f4"};
    } else {
        stats = (statistics_Configuration*)malloc(sizeof(statistics_Configuration));
        stats[0] = {simulations, timeBound, variable_threshhold, variable_id, isMax, isEstimate, loc_query};
    }

    for (int i = 0; i < NumberOfQueries; i++) {
        smc(conf, stats[i]);
    }


    return 1;
}




bool HandleCommandLineArguments(int argc, char **argv, string *filename, int *seed, statistics_Configuration* stats, bool CONST_QUERY) {
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
        } else if (arg == "-q" || arg == "--query") {
            CONST_QUERY = false;
            if (i + 7 < argc) {
                // statistics_Configuration
                /*int simulations = 1000;
                int timeBound = 100;
                int variable_threshhold = 10;
                int variable_id = 5;
                bool isMax = true; // Gather info on either the max value of the variable or the min
                bool isEstimate = True;
                string loc_query = "";*/
                std::string sims_string = argv[i + 1];
                std::string timb_string = argv[i + 2];
                std::string var_thresh_string = argv[i + 3];
                std::string var_id_string = argv[i + 4];
                std::string is_Max_string = argv[i + 5];
                std::string is_Estimate_string = argv[i + 6];
                stats->loc_query = argv[i + 7];


                i++; // Skip the next argument
                try {
                    stats->simulations = std::stoi(sims_string);
                    stats->timeBound = std::stoi(timb_string);
                    stats->variable_threshhold = std::stoi(var_thresh_string);
                    stats->variable_id = std::stoi(var_id_string);
                    stats->isMax = static_cast<bool>(std::stoi(is_Max_string));
                    stats->isEstimate = static_cast<bool>(std::stoi(is_Estimate_string));
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
        } else if (arg == "-h" || arg == "--help") {
            cout << "Use -m or --model, followed by a path, for inputting a path the the model XML file." << endl;
            cout << "Use -s or --seed, followed by a number, to initialize curand with a constant seed. (0 = random seed)" << endl;
            return false;
        }
        else {
            std::cerr << "Error: unknown option: " << arg << std::endl;
            return false;
        }
    }
    return true;
}
