// #include <iostream>
// #include <vector>
// #include <string>
// #include <memory>
// #include <stdexcept>
//
// #include "uppaal_xml_parser.h"
// #include <iostream>
//
// int main() {
//     uppaal_xml_parser parser;
//
//     const char* file_path = "./UppaalModelSamples/ball.xml";
//
//     try {
//         network parsed_network = parser.parse_xml(file_path);
//
//         // Here you would typically do something with the parsed network
//         // For example, print some information about it
//         std::cout << "Parsing successful. Network contains "
//                   << parsed_network.nodes.size() << " nodes." << std::endl;
//     }
//     catch (const std::exception& e) {
//         std::cerr << "Error parsing XML: " << e.what() << std::endl;
//         return 1;
//     }
//
//     return 0;
// }

#include <chrono>

#include "engine/Domain.h"
#include "automata_parser/uppaal_xml_parser.h"
#include <exception>
#include <iostream>

#include "automata_parser/instantiate_parser.h"

struct network_props
{
    std::chrono::steady_clock::time_point pre_optimisation_start;
    std::chrono::steady_clock::time_point post_optimisation_start;
    std::unordered_map<int, std::string>* variable_names;
    std::unordered_map<int, std::string>* node_names;
    std::unordered_map<int, std::string>* template_names;
    std::unordered_map<int, int>* node_network;
    std::unordered_map<int, node*> node_map;
};

int main()
{
    // Hardcoded path to the XML file
    std::string filename = "../automata_parser/XmlFiles/UppaalBehaviorTest.xml";

    abstract_parser* parser = instantiate_parser(filename);

    network net = parser->parse(filename);
    std::cout << "Parsing successful. Network details:" << std::endl;

    network_props properties = {};

    properties.node_names = new std::unordered_map<int, std::string>(*parser->get_nodes_with_name());
    properties.node_network = new std::unordered_map<int, int>(*parser->get_subsystems());
    properties.variable_names = new std::unordered_map<int, std::string>(*parser->get_clock_names()); // this can create mem leaks.
    properties.template_names = new std::unordered_map<int, std::string>(*parser->get_template_names());

    std::cout << "TEST";


    return 0;
}

#include <crt/host_defines.h>

#include "automata_parser/uppaal_xml_parser.h"
#include "engine/Domain.h"

/*
__host__ network uppaal_xml_parser::parse(const string& file)
{
    try
    {
        return parse_xml(file.c_str());
    }
    catch (const std::runtime_error &ex)
    {
        throw runtime_error(ex.what());
    }
}
*/

