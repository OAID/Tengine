#ifndef __GRU_HPP__
#define __GRU_HPP__

#include "operator.hpp"
#include "gru_param.hpp"

namespace TEngine {

class GRU : public OperatorWithParam<GRU, GRUParam>
{
public:
    GRU(void)
    {
        name_ = "GRU";
    }
    GRU(const GRU&) = default;
    void SetSchema(void) override;
    bool InferShape(const std::vector<TShape>&, std::vector<TShape>&, int layout) override;
    const char* GetBiasName(void)
    {
        return "gates/bias";
    }
    const char* GetKernelName(void)
    {
        return "gates/kernel";
    }
    const char* GetInitHiddenName(void)
    {
        return "init_h";
    }
    const char* GetCandidateKernelName(void)
    {
        return "candidate/kernel";
    }
    const char* GetCandidateBiasName(void)
    {
        return "candidate/bias";
    }
    const char* Geti2hweightName(void)
    {
        return "i2h_weight";
    }
    const char* Geti2hbiasName(void)
    {
        return "i2h_bias";
    }
    const char* Geth2hweightName(void)
    {
        return "h2h_weight";
    }
    const char* Geth2hbiasName(void)
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