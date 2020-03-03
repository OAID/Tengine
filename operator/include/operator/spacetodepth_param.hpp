#ifndef __SPACETODEPTH_PARAM_HPP__
#define __SPACETODEPTH_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine
{

struct SpaceToDepthParam : public NamedParam
{
    int block_size;
    /* data */  
    DECLARE_PARSER_STRUCTURE(SpaceToDepthParam)
    {
        DECLARE_PARSER_ENTRY(block_size);
    }
};


} // namespace TEngine
#endif