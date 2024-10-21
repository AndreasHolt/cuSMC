#include "declaration_parser.h"
#include "string_extractor.h"
#include "arithmetic-expression-evaluator.h"

using namespace std;
using namespace helper;
using namespace evalguy;

list<declaration> declaration_parser::parse_keyword(const string& lines, declaration_types type)
{
    list<declaration> result;
    const string keyword = decl_type_map_.at(type);

    for (const auto& line : split_expr(lines, ','))
    {
        const extract_declaration extracted_declaration = string_extractor::extract(extract_declaration(line, keyword));

        const string expr = extracted_declaration.right.empty() ? "" : to_string(eval_expr(extracted_declaration.right, &local_vars_, const_global_vars_));
    
        for (const string& extracted_keyword : extracted_declaration.keywords)
        {
            if (!expr.empty() && (type == clock_type || type == chan_type))
            {
                result.emplace_back(type, extracted_keyword, expr, type == clock_type ? global_clock_id_counter_++ : global_chan_id_counter_++);
            }
            else if (type == clock_type || type == chan_type)
                result.emplace_back(type, extracted_keyword, "0", type == clock_type ? global_clock_id_counter_++ : global_chan_id_counter_++);
            else if (!expr.empty())
                result.emplace_back(type, extracted_keyword, expr, 1);
            else
                result.emplace_back(type, extracted_keyword, "0", 1);

            local_vars_.emplace(extracted_keyword, !expr.empty() ? stof(expr) : 0);
        }
    }
    
    return result;
}

bool is_this_keyword(const string& expr, const string& keyword)
{
    return expr.substr(0, keyword.length()) == keyword;
}


void declaration_parser::number_parser(const string& input_string, list<declaration>* result, bool is_const = false)
{
    if (is_this_keyword(input_string,"double"))
    {
        list<declaration> cloc_decls = parse_keyword(input_string, is_const ? const_double_type : double_type);
        result->insert(result->end(), cloc_decls.begin(), cloc_decls.end());
    }
    if (is_this_keyword(input_string,"bool"))
    {
        list<declaration> cloc_decls = parse_keyword(input_string, is_const ? const_bool_type : bool_type);
        result->insert(result->end(), cloc_decls.begin(), cloc_decls.end());
    }
    if (is_this_keyword(input_string,"int"))
    {
        list<declaration> cloc_decls = parse_keyword(input_string, is_const ? const_int_type : int_type);
        result->insert(result->end(), cloc_decls.begin(), cloc_decls.end());
    }
}


list<declaration> declaration_parser::parse(const string& decl, unordered_map<string,double>* const_global_vars)
{
    this->const_global_vars_ = const_global_vars;
    
    const list<string> lines = split_expr(decl, '\n');
    list<declaration> result;

    for (const auto& line : lines)
    {
        if (line.empty())
            continue;
        
        string line_trimmed = remove_while(line, ' ');
        //Remove tabs
        line_trimmed = remove_while(line, '\t');
        
        if (line_trimmed.substr(0,2) == "//")
            continue;

        if (is_this_keyword(line_trimmed,"clock"))
        {
            list<declaration> cloc_decls = parse_keyword(line_trimmed, clock_type);
            result.insert(result.end(), cloc_decls.begin(), cloc_decls.end());
        }
        else if (is_this_keyword(line_trimmed,"const"))
        {   
            string const_string = remove_while(line_trimmed.substr(5), ' ');
            number_parser(const_string, &result, true);
        }
        else if (is_this_keyword(line_trimmed, "broadcastchan"))
        {
            list<declaration> chan_decls = parse_keyword(line_trimmed, chan_type);
            result.insert(result.end(), chan_decls.begin(), chan_decls.end());
        }
        number_parser(line_trimmed, &result);
    }
    
    return result;
}
