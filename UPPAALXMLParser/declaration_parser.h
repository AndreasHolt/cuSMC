#pragma once
#ifndef DECLARATION_PARSER
#define DECLARATION_PARSER
#include "helper_methods.h"
#include <list>
#include <string>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include "declaration.h"
#include "declaration_parser.h"

using namespace std;



class declaration_parser
{
private:
    int global_clock_id_counter_ = 0;
    int global_chan_id_counter_ = 0;
    unordered_map<string,double> local_vars_ = unordered_map<string,double>();
    unordered_map<string,double>* const_global_vars_ = nullptr;
    list<declaration> parse_keyword(const string& lines, declaration_types type);
    void number_parser(const string& input_string, list<declaration>* result, bool is_const);
    string val_;
    const map<declaration_types, string> decl_type_map_ { {const_bool_type, "bool"}, {bool_type, "bool"}, {const_double_type, "double"},{const_int_type, "int"},{clock_type,"clock"}, {double_type,"double"}, {int_type, "int"}, {chan_type, "broadcastchan"}};
public:
    std::list<declaration> parse(const std::string& decl, unordered_map<string,double>* const_global_vars);
};

#endif