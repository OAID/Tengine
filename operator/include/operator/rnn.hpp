#ifndef __RNN_HPP__
#define __RNN_HPP__

#include "operator.hpp"
#include "rnn_param.hpp"

namespace TEngine {

class RNN : public OperatorWithParam<RNN, RNNParam>
{
public:
    RNN(void)
    {
        name_ = "RNN";
    }
    RNN(const RNN&) = default;
    void SetSchema(void) override;
    bool InferShape(const std::vector<TShape>&, std::vector<TShape>&, int layout) override;
    const char* GetBiasName(void)
    {
        return "bias";
    }
    const char* GetInitHiddenName(void)
    {
        return "init_h";
    }
};

}    // namespace TEngine

#endif