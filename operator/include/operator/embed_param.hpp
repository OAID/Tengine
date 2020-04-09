#ifndef __EMBED_PARAM_HPP__
#define __EMBED_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct EmbedParam : public NamedParam
{
    int num_output;
    int input_dim;
    int bias_term;//if use bias
    int weight_data_size;

    DECLARE_PARSER_STRUCTURE(EmbedParam)
    {
        DECLARE_PARSER_ENTRY(num_output);
        DECLARE_PARSER_ENTRY(input_dim);
        DECLARE_PARSER_ENTRY(bias_term);
        DECLARE_PARSER_ENTRY(weight_data_size);
    };
};
}
#endif
