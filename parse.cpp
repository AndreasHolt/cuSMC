#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

#include "uppaal_xml_parser.h"
#include <iostream>

int main() {
    uppaal_xml_parser parser;
    
    const char* file_path = "./UppaalModelSamples/ball.xml";
    
    try {
        network parsed_network = parser.parse_xml(file_path);
        
        // Here you would typically do something with the parsed network
        // For example, print some information about it
        std::cout << "Parsing successful. Network contains " 
                  << parsed_network.nodes.size() << " nodes." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error parsing XML: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
