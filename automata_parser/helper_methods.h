#pragma once
#ifndef HELPER
#define HELPER
#include <list>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "parser_exception.h"
#include <regex>

using namespace std;

namespace helper
{
    inline bool string_contains(const string& expr, const string& element)
    {
        return expr.find(element)!=std::string::npos;
    }
    
    inline std::list<string> split_all(const string& input, const string& delimiter){
        std::list<string> result;
        size_t pos = 0;
        string s = input;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            std::string token = s.substr(0, pos);
            result.push_back(token);
            s.erase(0, pos + delimiter.length());
        }
        result.push_back(s);
        return result;
    }
    
    inline string take_after(const string& s, const string& while_string)
    {
        return s.substr(s.find(while_string)+while_string.length());
    }

    inline string take_while(const string& s, const string& while_string)
    {
        return s.substr(0,s.find(while_string));
    }

    inline string remove_while(const string& s, const char while_char)
    {
        for (size_t i = 0; i < s.length(); i++)
        {
            if (s[i] != while_char)
                return s.substr(i);
        }

        THROW_LINE("how: " + s + " \n WITH THE WHILE_CHAR: " + while_char)
    }
    
    inline string replace_all(std::string str, const std::string& from, const std::string& to) {
        size_t start_pos = 0;
        while((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
        }
        return str;
    }

    inline bool in_array(const char &value, const vector<char> &array)
    {
        return std::find(array.begin(), array.end(), value) != array.end();
    } 
    
    inline string remove_whitespace(const string& expr)
    {
        return replace_all(expr, " ", "");
    }

    inline list<std::string> split_expr(const string& expr)
    {
        list<string> result;
        std::stringstream test(expr);
        std::string segment;
        while(std::getline(test, segment, '&'))
        {
            string s = replace_all(segment, "&", "");
            std::regex e ("random\\((.*)\\)"); 
            s = std::regex_replace (s,e,"random|$1|");
            s = helper::replace_all(s, "(", "");
            s = helper::replace_all(s, ")", "");
            std::regex e2 ("random\\|(.*)\\|");
            s = std::regex_replace (s,e2,"random($1)");
            result.push_back(s);
        }
        return result;
    }

    inline list<std::string> split_expr(const string& expr, const char split_on)
    {
        list<string> result;
        std::stringstream test(expr);
        std::string segment;
        while(std::getline(test, segment, split_on))
        {
            result.push_back(segment);
        }
        return result;
    }
}

#endif

