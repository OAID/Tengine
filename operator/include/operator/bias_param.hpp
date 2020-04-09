#ifndef __BIAS_PARAM_HPP__
#define __BIAS_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct BiasParam : public NamedParam
{
    int bias_size;

    DECLARE_PARSER_STRUCTURE(BiasParam)
    {
        DECLARE_PARSER_ENTRY(bias_size);
    }
};
}    // namespace TEngine

#endif
