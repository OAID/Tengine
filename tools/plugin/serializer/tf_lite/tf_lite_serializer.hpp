#ifndef __TF_LITE_SERIALIZER_HPP__
#define __TF_LITE_SERIALIZER_HPP__

#include "flatbuffers/flatbuffers.h"
#include "schema_generated.h"

#include "serializer.hpp"

using TFLiteTensor = ::tflite::Tensor;
using TFLiteOperator = ::tflite::Operator;
using LiteModel = ::tflite::Model;

namespace TEngine {

class TFLiteSerializer : public Serializer
{
public:
    struct LiteTensor;

    struct LiteNode
    {
        int idx;
        std::string op;
        std::string name;
        std::vector<LiteTensor*> inputs;
        std::vector<LiteTensor*> outputs;

        const TFLiteOperator* lite_op;

        StaticNode* static_node;
    };

    struct LiteTensor
    {
        int idx;
        std::string name;
        std::string type;
        std::vector<int> shape;

        StaticTensor* static_tensor;
        LiteNode* producer;
        const TFLiteTensor* tf_tensor;
        bool graph_input;
        bool graph_output;

        LiteTensor(void)
        {
            tf_tensor = nullptr;
            static_tensor = nullptr;
            producer = nullptr;
            graph_input = false;
            graph_output = false;
        }
    };

    struct LiteGraph
    {
        std::vector<LiteNode*> seq_nodes;
        std::vector<LiteTensor*> input_tensors;
        std::vector<LiteTensor*> output_tensors;
        std::vector<LiteTensor*> tensor_list;

        const LiteModel* lite_model;

        ~LiteGraph(void)
        {
            for(auto node : seq_nodes)
                delete node;
        }
    };

    bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* static_graph) override;

    unsigned int GetFileNum(void) final
    {
        return 1;
    }

    bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) override
    {
        return false;
    }
    bool LoadConstTensor(int fd, StaticTensor* const_tensor) override
    {
        return false;
    }

    TFLiteSerializer(void)
    {
        name_ = "TFLite";
    }

protected:
    bool LoadModelFromMem(char* mem_addr, int mem_size, StaticGraph* graph);

    bool ConstructGraph(const LiteModel* tf_model, LiteGraph* lite_graph);

    bool OptimizeGraph(LiteGraph* lite_graph);

    bool GenerateStaticGraph(LiteGraph* lite_graph, StaticGraph* graph);
    bool LoadTensorScaleAndZero(StaticTensor* static_tensor, LiteTensor* lite_tensor);

    void DumpLiteGraph(LiteGraph* lite_graph);
    void DumpLiteTensor(LiteTensor* tensor);

    bool LoadConstLiteTensor(StaticTensor* static_tensor, LiteTensor* tensor, LiteGraph* lite_graph,
                             StaticGraph* graph);
    bool LoadLiteTensor(LiteTensor* tensor, LiteGraph* lite_graph, StaticGraph* graph);
    bool LoadLiteNode(LiteNode* node, LiteGraph* lite_graph, StaticGraph* graph);

    void CreateGraphInputNode(LiteTensor* tensor, StaticGraph* graph);
};

}    // namespace TEngine

#endif
