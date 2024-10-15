#pragma once

#ifndef DECLARATION_H
#define DECLARATION_H

#include <string>

using namespace std;

enum declaration_types {clock_type, const_bool_type, bool_type, const_double_type, const_int_type, chan_type, double_type, int_type};
#define THROW_LINE(arg); throw parser_exception(arg, __FILE__, __LINE__);

class declaration
{
private:
    declaration_types type_;
    string var_name_;
    double value_;
    int id_;
public:
    declaration(const declaration_types type, const string& var_name, const string& val, const int id)
    {
        this->type_ = type;
        this->var_name_ = var_name;
        this->value_ = stod(val);
        this->id_ = id;
    }
    
    declaration_types get_type() const
    {
        return this->type_;
    }
    
    string get_name() const
    {
        return this->var_name_;
    }

    double get_value() const
    {
        return this->value_;
    }

    int get_id() const
    {
        return this->id_;
    }
};
#endif