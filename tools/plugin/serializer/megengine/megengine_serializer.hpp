
#ifndef __MEGENGINE_SERIALIZER_HPP__
#define __MEGENGINE_SERIALIZER_HPP__

#include <iostream>

#include "logger.hpp"
#include "serializer.hpp"
#include "megbrain/serialization/serializer.h"

namespace mgb {
namespace cg {
// declaration of impl class to access its methods
class ComputingGraphImpl : public ComputingGraph
{
public:
    std::vector<std::unique_ptr<OperatorNodeBase>>&& all_oprs() const;
    const OprNodeArray& var_receiver(VarNode* var) const;
};
}    // namespace cg
}    // namespace mgb

namespace TEngine {

class MegengineSerializer : public Serializer
{
public:
    MegengineSerializer()
    {
        name_ = "megengine_loader";
        version_ = "0.1";
        format_name_ = "megengine";
    }

    virtual ~MegengineSerializer() {}

    unsigned int GetFileNum(void) override
    {
        return 1;
    }

    bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* static_graph) override;

    bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) override
    {
        return false;
    }

    bool LoadConstTensor(int fd, StaticTensor* const_tensor) override
    {
        return false;
    }

protected:
    bool LoadGraph(mgb::cg::ComputingGraphImpl* cg, StaticGraph* graph);
    bool LoadNode(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase* mge_op);
};

}    // namespace TEngine

#endif