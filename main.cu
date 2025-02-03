//
// Created by andwh on 04/11/2024.
//

#include "main.cuh"
#include "smc.cuh"
#include "automata_parser/uppaal_xml_parser.h"
#include <regex>

int main(int argc, char *argv[]) {

    std::string filename = "../xml_files/LargeModels/aloha_50.xml";
    int curand_seed = 1235;

    // Statistics
    statistics_Configuration stat_input = statistics_Configuration();
    stat_input.simulations = 16384;
    stat_input.timeBound = 100;
    stat_input.variable_threshhold = -1;
    stat_input.variable_id = 1;
    stat_input.isMax = true; // Gather info on either the max value of the variable or the min
    stat_input.isEstimate = true;
    stat_input.loc_query = "";



    bool cliSucceeded = HandleCommandLineArguments(argc, argv, &filename, &curand_seed, &stat_input);
    if (!cliSucceeded) {return 0;}

    const configuration conf = {filename, curand_seed};

    smc(conf, stat_input);


    return 1;
}


bool HandleCommandLineArguments(int argc, char **argv, string *filename, int *seed, statistics_Configuration* stats) {
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
            if (i + 7 < argc) {
                std::string sims_string = argv[++i];
                std::string timb_string = argv[++i];
                std::string var_thresh_string = argv[++i];
                std::string var_id_string = argv[++i];
                std::string is_Max_string = argv[++i];
                std::string is_Estimate_string = argv[++i];
                stats->loc_query = argv[++i];

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
            cout << "Use -q or --query, followed by : number of simulation, time bound, a variable threshold, a variable name (id), Max=1 or Min=0, Estimate=1 or Probability=0." << endl;
            return false;
        }
        else {
            std::cerr << "Error: unknown option: " << arg << std::endl;
            return false;
        }
    }
    return true;
}
