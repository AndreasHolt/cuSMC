#pragma once
#ifndef STRING_EXTRACTOR
#define STRING_EXTRACTOR
#include <iostream>
#include <string>
#include <utility>
#include "helper_methods.h"
using namespace std;
using namespace helper;
struct extract_input
{
    const string input;
    
public:
    explicit extract_input(const string& input) : input(remove_whitespace(input))
    {
    }

    
};

struct extract_if_statement : extract_input
{
    const string condition;
    const string if_true;
    const string if_false;
    
    explicit extract_if_statement(const string& ex)
        : extract_input(ex), condition(), if_true(), if_false()
    {
    }

    explicit extract_if_statement(string condition, string if_true, string if_false, const string& trimmed_input) : extract_input(trimmed_input),
        condition(std::move(condition)), if_true(std::move(if_true)), if_false(std::move(if_false))
    {
    }
};

struct extract_condition : extract_input
{
    const string op;
    const string left;
    const string right;
    
    explicit extract_condition(const string& ex)
        : extract_input(ex), op(), left(), right()
    {
    }

    explicit extract_condition(string op, const basic_string<char>& left, string right, const string& trimmed_input)
        : extract_input(trimmed_input), op(std::move(op)), left(left), right(std::move(right))
    {
    }
};

struct extract_assignment : extract_input
{
    const string left;
    const string right;
    
    explicit extract_assignment(const string& ex)
        : extract_input(ex), left(), right()
    {
    }

    explicit extract_assignment(const basic_string<char>& left, string right, const string& ex)
        : extract_input(ex), left(left), right(std::move(right))
    {
    }
};

struct extract_sync : extract_input
{
    const bool is_listener;
    const string keyword;
    
    explicit extract_sync(const string& ex)
            : extract_input(ex), is_listener(), keyword()
        {
    }
    
     explicit extract_sync(const bool is_listener, string keyword,const string& ex)
        : extract_input(ex),
          is_listener(is_listener),
          keyword(std::move(keyword))
    {
    }

};

struct extract_probability : extract_input
{
    const string value;
    explicit extract_probability(const string& ex)
       : extract_input(ex),
         value()
    {
    }
    
    explicit extract_probability(string value, const string& ex)
        : extract_input(ex),
          value(std::move(value))
    {
    }
};

struct extract_node_id : extract_input
{
    const int id;

    extract_node_id(const string& ex)
        : extract_input(ex),
          id()
    {
    }
    
    extract_node_id(const int id, const string& ex)
        : extract_input(ex),
          id(id)
    {
    }
};

struct extract_declaration : extract_input
{
    string input_keyword;
    list<string> keywords;
    const string right;
    
    explicit extract_declaration(const string& ex, string input_keyword)
        : extract_input(ex), input_keyword(std::move(input_keyword))
    {
    }

    explicit extract_declaration(list<string> keywords, string right,const string& ex)
        : extract_input(ex), keywords(std::move(keywords)), right(std::move(right))
    {
    }
};

class string_extractor
{
public:
    static extract_if_statement extract(const extract_if_statement& extract);
    static extract_condition extract(const extract_condition& extract);
    static extract_assignment extract(const extract_assignment& extract);
    static extract_sync extract(const extract_sync& extract);
    static extract_probability extract(const extract_probability& extract);
    static extract_declaration extract(const extract_declaration& extract);
    static int extract(const extract_node_id& extract);
};

#endif