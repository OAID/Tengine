#ifndef __COMPARISON_PARAM_HPP__
#define __COMPARISON_PARAM_HPP__

#include "parameter.hpp"

enum CompType
{
    COMP_EQUAL,
    COMP_NOT_EQUAL,
    COMP_GREATER,
    COMP_GREATER_EQUAL,
    COMP_LESS,
    COMP_LESS_EQUAL
};

namespace TEngine{

struct ComparisonParam : public NamedParam
{
    int type;

    DECLARE_PARSER_STRUCTURE(ComparisonParam)
    {
        DECLARE_PARSER_ENTRY(type);
    };
};

}
#endif