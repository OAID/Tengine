#ifndef __GATHER_PARAM_HPP__
#define __GATHER_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct GatherParam : public NamedParam
{
    int axis;
    int indices_num;
    bool is_onnx;
    DECLARE_PARSER_STRUCTURE(GatherParam)
    {
        DECLARE_PARSER_ENTRY(axis);
        DECLARE_PARSER_ENTRY(indices_num);
        DECLARE_PARSER_ENTRY(is_onnx);
    };
};

}    // namespace TEngine

#endif
