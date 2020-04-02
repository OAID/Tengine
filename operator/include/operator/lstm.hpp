#ifndef __LSTM_HPP__
#define __LSTM_HPP__

#include "operator.hpp"
#include "lstm_param.hpp"

namespace TEngine {

class LSTM : public OperatorWithParam<LSTM, LSTMParam>
{
public:
    LSTM(void)
    {
        name_ = "LSTM";
    }
    LSTM(const LSTM&) = default;
    void SetSchema(void) override;
    bool InferShape(const std::vector<TShape>&, std::vector<TShape>&, int layout) override;
    const char* GetKernelName(void)
    {
        return "kernel";
    }
    const char* GetBiasName(void)
    {
        return "bias";
    }
    const char* GetProjectionName(void)
    {
        return "projection";
    }
    const char* GetPeepholeForgetName(void)
    {
        return "w_f_diag";
    }
    const char* GetPeepholeInputName(void)
    {
        return "w_i_diag";
    }
    const char* GetPeepholeOutputName(void)
    {
        return "w_o_diag";
    }
    const char* GetInitCellName(void)
    {
        return "init_c";
    }
    const char* GetInitHiddenName(void)
    {
        return "init_h";
    }
    const char* Geti2hKernelName(void)
    {
        return "i2h_weight";
    }
    const char* Geti2hBiasName(void)
    {
        return "i2h_bias";
    }
    const char* Geth2hKernelName(void)
    {
        return "h2h_weight";
    }
    const char* Geth2hBiasName(void)
    {
        return "h2h_bias";
    }
    const char* GetFusedKernelName(void)
    {
        return "parameters";
    }
};

}    // namespace TEngine

#endif
