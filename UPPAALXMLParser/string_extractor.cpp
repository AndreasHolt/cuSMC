#include "string_extractor.h"

string get_constraint_op(const string& expr)
{
    if(expr.find("<=") != std::string::npos)
        return "<=";
    if(expr.find(">=") != std::string::npos)
        return ">=";
    if(expr.find("==") != std::string::npos)
        return "==";
    if(expr.find("!=") != std::string::npos)
        return "!=";
    if(expr.find('<') != std::string::npos)
        return "<";
    if(expr.find('>') != std::string::npos)
        return ">";

    THROW_LINE("Operand in " + expr + " not found, sad..");
}

extract_if_statement string_extractor::extract(const extract_if_statement& extract)
{
    // condition?if_true:if_false;
    const string input = extract.input;
    const string condition = helper::replace_all(helper::take_while(input, "?"), " ", "");
    const string if_true = helper::replace_all(helper::take_while(helper::take_after(input, "?"), ":"), " ", "");
    const string if_false = helper::replace_all(helper::take_after(input, ":"), ";", "");
    return extract_if_statement(condition, if_true, if_false,input);
}

extract_condition string_extractor::extract(const extract_condition& extract)
{
    // left op right
    if (extract.input.empty())
        return extract_condition("","","","");
    
    const string condition = extract.input;
    const string op = get_constraint_op(condition);
    const string left = helper::take_while(condition, op);
    const string right = helper::take_after(condition, op);
    return extract_condition(op, left, right,condition);
}

extract_assignment string_extractor::extract(const extract_assignment& extract)
{
    // left_side=right_side or left_side:=right_side
    const string left_side = helper::string_contains(extract.input, ":=") ? helper::take_while(extract.input, ":=") : helper::take_while(extract.input, "=");
    string right_side_of_equal = helper::take_after(extract.input, "=");
    right_side_of_equal = helper::replace_all(right_side_of_equal, "true", "1");
    right_side_of_equal = helper::replace_all(right_side_of_equal, "false", "0");
    return extract_assignment(left_side,right_side_of_equal,extract.input);
}

extract_sync string_extractor::extract(const extract_sync& extract)
{
    const bool is_listener = extract.input.find('?')!=std::string::npos;
    const string sync_keyword = helper::replace_all(extract.input, is_listener ? "?" : "!", "");
    return extract_sync(is_listener, sync_keyword, extract.input);
}

extract_probability string_extractor::extract(const extract_probability& extract)
{
    return extract_probability(extract.input, extract.input);
}

int string_extractor::extract(const extract_node_id& extract)
{
    string result = extract.input;
    return stoi(result.replace(0,2,""))+1;
}

extract_declaration string_extractor::extract(const extract_declaration& extract)
{
    string input_string = extract.input;
    input_string = helper::replace_all(input_string, extract.input_keyword, "");
    input_string = helper::replace_all(input_string, ";", "");
    const string left = helper::string_contains(input_string, ":=") ? helper::take_while(input_string, ":=") 
        : helper::string_contains(input_string, "=") ?  helper::take_while(input_string, "=") : input_string;
    const std::list<string> keywords = helper::split_all(left, ",");
    
    string right = helper::string_contains(input_string, "=") ? helper::take_after(input_string, "=") : "";
    right = helper::replace_all(right, "true", "1");
    right = helper::replace_all(right, "false", "0");
    return extract_declaration(keywords, right, extract.input);
}