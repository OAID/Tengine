#ifndef __NOOP_HPP__
#define __NOOP_HPP__

#include "operator.hpp"
namespace TEngine{

class Noop : public OperatorNoParam<Noop>
{
public:
    Noop(void)
    {
        name_ = "Noop";
    }

    Noop(const Noop&) = default;

    bool InferShape(const std::vector<TEngine::TShape>&, std::vector<TEngine::TShape>&, int layout) override;
    void SetSchema(void) override;
};

}
#endif