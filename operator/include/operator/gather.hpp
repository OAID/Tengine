#ifndef __GATHER_HPP__
#define __GATHER_HPP__

#include "operator.hpp"
#include "gather_param.hpp"

namespace TEngine {

class Gather : public OperatorWithParam<Gather, GatherParam>
{
public:
    Gather()
    {
        name_ = "Gather";
    }
    Gather(const Gather& src) = default;
    virtual ~Gather(){};
    bool InferShape(const std::vector<TEngine::TShape>&, std::vector<TEngine::TShape>&, int) override;
    void SetSchema(void) override;
};

}    // namespace TEngine

#endif