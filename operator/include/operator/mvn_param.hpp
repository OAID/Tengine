#ifndef __MVN_PARAM_HPP__
#define __MVN_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct MVNParam : public NamedParam
{
    int normalize_variance;
    int across_channels;
    float eps;

    DECLARE_PARSER_STRUCTURE(MVNParam)
    {
        DECLARE_PARSER_ENTRY(normalize_variance);
        DECLARE_PARSER_ENTRY(across_channels);
        DECLARE_PARSER_ENTRY(eps);
    };
};

}

#endif