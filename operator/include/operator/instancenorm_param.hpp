#ifndef __INSTANCENORM_PARAM_HPP__
#define __INSTANCENORM_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine{

struct InstanceNormParam : public NamedParam
{
    float eps;
    DECLARE_PARSER_STRUCTURE(InstanceNormParam)
    {
        DECLARE_PARSER_ENTRY(eps);
    };
};

}
#endif