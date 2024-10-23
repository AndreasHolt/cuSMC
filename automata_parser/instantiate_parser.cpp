//
// Created by ucloud on 10/23/24.
//

#include "instantiate_parser.h"
#include "abstract_parser.h"
#include "uppaal_xml_parser.h"

inline bool str_ends_with(const std::string& full_string, const std::string& ending) {
    if (ending.size() > full_string.size()) {
        return false;  // Early return if ending is longer than full_string
    }
    return full_string.substr(full_string.size() - ending.size()) == ending;
}

abstract_parser* instantiate_parser(const std::string& filepath)
{
    if(str_ends_with(filepath, ".SMAcc"))
    {
        throw std::logic_error("Parsing of .comp files not support yet");
    }
    else if (str_ends_with(filepath, ".xml"))
    {
        return new uppaal_xml_parser();
    }
    else
    {
        throw std::runtime_error(R"(Could not parse file. Should end in ".SMAcc" or ".xml", but neither was detected)");
    }
}
