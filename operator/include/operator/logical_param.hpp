#ifndef __LOGICAL_PARAM_HPP__
#define __LOGICAL_PARAM_HPP__

#include "parameter.hpp"

enum LogicalType
{
    LOGICAL_AND,
    LOGICAL_OR,
};

namespace TEngine {

struct LogicalParam : public NamedParam
{
    // std::string method;
    // LogicalType type;
    int type;

    DECLARE_PARSER_STRUCTURE(LogicalParam)
    {
        DECLARE_PARSER_ENTRY(type);
    };
};

}    // namespace TEngine

#endif