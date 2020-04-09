#ifndef __THRESHOLD_PARAM_HPP__
#define __THRESHOLD_PARAM_HPP__

#include "parameter.hpp"
namespace TEngine{

struct ThresholdParam : public NamedParam
{
    float threshold;

    DECLARE_PARSER_STRUCTURE(ThresholdParam)
    {
        DECLARE_PARSER_ENTRY(threshold);
    }
};


}
#endif