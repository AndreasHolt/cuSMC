//
// Created by ucloud on 10/23/24.
//

#ifndef OUTPUT_H
#define OUTPUT_H

#include "output.h"

#include <chrono>
#include <unordered_map>
#include <string>
#include "engine/Domain.h"

struct network_props {
    std::chrono::steady_clock::time_point pre_optimisation_start;
    std::chrono::steady_clock::time_point post_optimisation_start;
    std::unordered_map<int, std::string> *variable_names;
    std::unordered_map<int, std::string> *node_names;
    // std::unordered_map<int, std::string>* template_names;
    std::unordered_map<int, int> *node_network;
    std::unordered_map<int, node *> node_map;

    // Our version
    std::unordered_map<int, list<edge> > *node_edge_map;
    std::list<int> *start_nodes;
    std::unordered_map<int, std::string> *template_names;
};


#endif //OUTPUT_H
