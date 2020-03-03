#ifndef __EMBED_HPP__
#define __EMBED_HPP__

#include "operator.hpp"
#include "embed_param.hpp"

namespace TEngine{

class Embed : public OperatorWithParam<Embed, EmbedParam>
{
public:
    Embed(void)
    {
        name_ = "Embedding";
  
    }
    Embed(const Embed&) = default;
    void SetSchema(void) override;
    bool InferShape(const std::vector<TShape>&, std::vector<TShape>&, int layout) override;
};

}
#endif