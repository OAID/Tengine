#include "operator/gru.hpp"
#include "operator/gru_param.hpp"
#include "static_graph.hpp"

namespace TEngine {

bool GRU::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    // input tensors:
    // 0 --- input: [seq_length, batch_size,input_size]
    // 1 --- kernel [ (input_size+hidden_size),hidden_state_size]
    // others: optional

    // output tensor: [output_len,batch_size,hidden_size]

    const TShape input_shape = ishape[0];
    int batch_size = input_shape.Shape(0);
    std::vector<int> dims(3);
    if(param_.mxnet_flag == 1)
    {
        batch_size = input_shape.Shape(1);
        dims[0] = input_shape.Shape(0);
        dims[1] = batch_size;
        dims[2] = param_.hidden_size;
    }
    else
    {
        dims[1] = input_shape.Shape(0);
        dims[0] = batch_size;
        dims[2] = param_.hidden_size;
    }

    oshape[0].SetDim(dims);

    // std::cout<<dims[0]<<","<< dims[1]<<","<<dims[2]<<"\n";

    return true;
}

void GRU::SetSchema(void)
{
    Input({"input:float32", "kernel:float32", "bias:float32", "init_h:float32"})
        .Output({"output:float32"})
        .SetAttr("clip", 0.0f)
        .SetAttr("output_len", 1)
        .SetAttr("sequence_len", 1)
        .SetAttr("input_size", 1)
        .SetAttr("hidden_size", 1)
        .SetAttr("has_clip", 0)
        .SetAttr("has_gate_bias", 0)
        .SetAttr("has_candidate_bias", 0)
        .SetAttr("has_init_state", 0)
        .SetDoc(R"DOC(GRU Cell
              input: input sequences, a 3D tensor [seq_length,batch_size,input_size]
              gate_kernel: gate weight tensor,[num_directions, hidden_size, ]
              gate_bias:   gate bias tensor, [num_directions, hidden_size]
              candidate_kernel: candidate weight tensor,[num_directions, hidden_size, ]
              candidate_bias:   candidate bias tensor, [num_directions, hidden_size]
              init_h: optional [hidden_size]
                 )DOC");
}
}    // namespace TEngine