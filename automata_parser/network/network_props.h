//
// Created by ucloud on 10/23/24.
//

#ifndef OUTPUT_H
#define OUTPUT_H

#include <chrono>
#include <unordered_map>
#include <string>
#include "include/engine/domain.h"
#include <list>

struct network_props {
    std::chrono::steady_clock::time_point pre_optimisation_start;
    std::chrono::steady_clock::time_point post_optimisation_start;
    std::unordered_map<int, std::string> *variable_names;
    std::unordered_map<int, std::string> *node_names;
    std::unordered_map<int, int> *node_network;
    std::unordered_map<int, node *> node_map;


    // Additional information needed to instantiate our coalesced model structure
    std::unordered_map<int, std::list<edge> > *node_edge_map;
    std::list<int> *start_nodes;
    std::unordered_map<int, std::string> *template_names;
    std::unordered_map<string, int> vars_map;
    std::unordered_map<std::string, int> template_name_int_map;
    std::unordered_map<std::string, int> node_name_int_map;

    network_props() : variable_names(nullptr),
                      node_names(nullptr),
                      node_network(nullptr),
                      node_edge_map(nullptr),
                      start_nodes(nullptr),
                      template_names(nullptr)
    {

    }

    ~network_props() {
        delete variable_names;
        delete node_names;
        delete node_network;
        delete node_edge_map;
        delete start_nodes;
        delete template_names;
    }
};

inline void populate_properties(network_props &properties, abstract_parser *parser) {
    if (!parser) {
        throw std::runtime_error("Null parser provided to populate_properties");
    }

    try {
        properties.node_edge_map = new std::unordered_map<int, std::list<edge> >(
            parser->get_node_edge_map());

        properties.start_nodes = new std::list<int>(
            parser->get_start_nodes());

        properties.template_names = new std::unordered_map<int, std::string>(
            *parser->get_template_names());

        properties.variable_names = new std::unordered_map<int, std::string>(
            *parser->get_clock_names());

        properties.node_network = new std::unordered_map<int, int>(
            *parser->get_subsystems());

        properties.node_names = new std::unordered_map<int, std::string>(
            *parser->get_nodes_with_name());

        properties.pre_optimisation_start = std::chrono::steady_clock::now();

        properties.vars_map = parser->get_variables_names_to_ids_map();

        for (auto itr = properties.template_names->cbegin(); itr != properties.template_names->cend(); itr++) {
            properties.template_name_int_map.insert({itr->second, itr->first});
        }

        for (auto itr = properties.node_names->cbegin(); itr != properties.node_names->cend(); itr++) {
            properties.node_name_int_map.insert({itr->second, itr->first});
        }
    } catch (const std::exception &e) {
        // Clean up any already allocated memory before rethrowing
        delete properties.node_edge_map;
        delete properties.start_nodes;
        delete properties.template_names;
        delete properties.variable_names;
        delete properties.node_network;
        delete properties.node_names;

        throw std::runtime_error(std::string("Failed to populate properties: ") + e.what());
    }
}

#endif //OUTPUT_H
