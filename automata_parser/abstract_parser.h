#ifndef ABSTRACT_PARSER_H
#define ABSTRACT_PARSER_H

#pragma once
#include <unordered_map>
#include <unordered_set>

#include "helper_methods.h"
#include "../engine/Domain.h"

class abstract_parser
{
public:
    virtual ~abstract_parser() = default;
    virtual network parse(const std::string& file) = 0;
    virtual std::unordered_map<int, std::string>* get_nodes_with_name() = 0;
    virtual std::unordered_map<int, std::string>* get_clock_names() = 0;
    virtual std::unordered_map<int, int>* get_subsystems() = 0;
    virtual std::unordered_map<int, string>* get_template_names() = 0;

    static void naive_multiply_instantiation(network* n, const int multiplier)
    {
        if(multiplier <= 0)
        {
            throw std::runtime_error("Cannot multiply model with 0 or less");
        }
        if(multiplier == 1) return; //dont do anything

        const int size = n->automatas.size;
        const int new_size = size * multiplier;
        node** originals = n->automatas.store;

        node** store = static_cast<node**>(malloc(sizeof(node*)*new_size));

        for (int i = 0; i < multiplier; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                const int k = i * size + j;
                store[k] = originals[j];
            }   
        }

        n->automatas = arr<node*>{store, new_size};
    }
    
    static std::unordered_set<std::string>* parse_query(const std::string& query)
    {
        if(query.empty()) return new unordered_set<std::string>();
        if(query.find(' ') != std::string::npos)
            throw std::runtime_error("Query contains space");
        
        //p0.Node0,P1.Node2,P1.Node0
        const std::list<std::string> split = helper::split_all(query, ",");
        std::unordered_set<std::string>* set = new std::unordered_set<std::string>();
        for(auto& q : split)
        {
            if(std::count(q.begin(), q.end(), '.') != 1)
                throw std::runtime_error("Query contains section with more or less than 1 dot ("+ q +")");
            set->insert(q);
        }
        return set;
    }
    
    static bool try_parse_block_threads(const std::string& str, unsigned* out_blocks, unsigned* out_threads)
    {
        const std::list<std::string> split = helper::split_all(str, ",");
        if(split.size() != 2) return false;
        const string& blocks = split.front();
        const string& threads = split.back();

        try
        {
            *out_blocks  = static_cast<unsigned>(stoi(blocks));
            *out_threads = static_cast<unsigned>(stoi(threads));
        }
        catch(invalid_argument&)
        {
            return false;
        }
        return true;
    }

    static bool try_parse_units(const std::string& str, bool* is_time, double* value)
    {
        const char unit = str.back();
        if(unit != 's' && unit != 't') return false;
        *is_time = unit == 't';

        const string val = str.substr(0, str.size() - 1);

        try
        {
            *value = stod(val);
        }
        catch(invalid_argument&)
        {
            return false;
        }
        return true;
    }
};

#endif