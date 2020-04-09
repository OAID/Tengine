#ifndef __SPARSETODENSE_PARAM_HPP__
#define __SPARSETODENSE_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct SparseToDenseParam : public NamedParam
{
    int output_shape_size0;
    int output_shape_size1;
    int default_value;

    DECLARE_PARSER_STRUCTURE(SparseToDenseParam)
    {
        DECLARE_PARSER_ENTRY(output_shape_size0);
        DECLARE_PARSER_ENTRY(output_shape_size1);
        DECLARE_PARSER_ENTRY(default_value);
    }
};

}    // namespace TEngine

#endif