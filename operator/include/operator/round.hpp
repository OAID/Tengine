#ifndef __ROUND_HPP__
#define __ROUND_HPP__

#include "operator.hpp"

namespace TEngine {

class Round : public OperatorNoParam<Round>
{
public:
    Round()
    {
        name_ = "Round";
    }
    Round(const Round& src) = default;
    virtual ~Round(){};

    void SetSchema(void) override;

    bool InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout) override;
};

}   // namespace TEngine

#endif
