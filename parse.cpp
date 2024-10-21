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

#include "engine/Domain.h"
#include "automata_parser/uppaal_xml_parser.h"
#include <exception>
#include <iostream>


int main()
{
    // Hardcoded path to the XML file
    std::string filename = "../automata_parser/XmlFiles/UppaalBehaviorTest.xml";

    uppaal_xml_parser parser;

    try
    {
        network net = parser.parse(filename);
        std::cout << "Parsing successful. Network details:" << std::endl;
        // std::cout << "Number of processes: " << net.processes.size() << std::endl;
        // Add more details as needed
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing file: " << e.what() << std::endl;
        return 1;
    }

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

