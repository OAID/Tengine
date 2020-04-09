#ifndef __SPARSETODENSE_HPP__
#define __SPARSETODENSE_HPP__

#include "operator.hpp"
#include "sparsetodense_param.hpp"

namespace TEngine {

class SparseToDense : public OperatorWithParam<SparseToDense, SparseToDenseParam>
{
public:
    SparseToDense()
    {
        name_ = "SparseToDense";
    }
    SparseToDense(const SparseToDense& src) = default;

    virtual ~SparseToDense() {}
    bool InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape,
                    int layout) override;
    void SetSchema(void) override;
};

}    // namespace TEngine

#endif