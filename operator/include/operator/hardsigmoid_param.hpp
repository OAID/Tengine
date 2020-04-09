#ifndef __HARDSIGMOID_PARAM_HPP__
#define __HARDSIGMOID_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine{

struct HardsigmoidParam : public NamedParam 
{
    float alpha; 
    float beta;

    DECLARE_PARSER_STRUCTURE(HardsigmoidParam)
    {
        DECLARE_PARSER_ENTRY(alpha);
        DECLARE_PARSER_ENTRY(beta);
    };
};

}

#endif