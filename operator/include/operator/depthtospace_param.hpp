#ifndef __DEPTHTOSPACE_PARAM_HPP__
#define __DEPTHTOSPACE_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine
{

struct DepthToSpaceParam : public NamedParam
{
    int block_size;
    /* data */  
    DECLARE_PARSER_STRUCTURE(DepthToSpaceParam)
    {
        DECLARE_PARSER_ENTRY(block_size);
    }
};


} // namespace TEngine
#endif