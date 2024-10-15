#include <sstream>

#ifndef parser_exception_H
#define parser_exception_H

#define THROW_LINE(arg); throw parser_exception(arg, __FILE__, __LINE__);

using namespace std;

class parser_exception : public std::runtime_error {
    std::string msg;
public:
    parser_exception(const std::string &arg, const char *file, int line) :
    std::runtime_error(arg) {
        std::ostringstream o;
        o << file << ":" << line << ": PARSER ERROR: " << arg;
        msg = o.str();
    }
    ~parser_exception() throw() {}
    const char *what() const throw() {
        return msg.c_str();
    }
};

#endif

