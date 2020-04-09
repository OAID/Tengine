#include "operator/rnn.hpp"
#include "operator/rnn_param.hpp"
#include "static_graph.hpp"

namespace TEngine {

bool RNN::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    // input tensors:
    // 0 --- input: [seq_length, batch_size,input_size]
    // 1 --- kernel [ (input_size+hidden_size),hidden_state_size]
    // others: optional

    // output tensor: [output_len,batch_size,hidden_size]

    const TShape input_shape = ishape[0];

    int batch_size = input_shape.Shape(1);

    std::vector<int> dims(3);

    dims[0] = param_.output_len;
    dims[1] = batch_size;
    dims[2] = param_.hidden_size;

    oshape[0].SetDim(dims);

    return true;
}

void RNN::SetSchema(void)
{
    Input({"input:float32", "kernel:float32", "bias:float32", "init_h:float32"})
        .Output({"output:float32"})
        .SetAttr("clip", 0.0f)
        .SetAttr("output_len", 1)
        .SetAttr("sequence_len", 1)
        .SetAttr("input_size", 1)
        .SetAttr("hidden_size", 1)
        .SetAttr("has_clip", 0)
        .SetAttr("has_bias", 0)
        .SetAttr("has_init_state", 0)
        .SetAttr("activation", RNN_ACT_TANH)
        .SetDoc(R"DOC(LSTM Cell
              input: input sequences, a 3D tensor [seq_length,batch_size,input_size]
              kernel: gate weight tensor,[num_directions, hidden_size, ]
              bias:   gate bias tensor, [num_directions, hidden_size]
              init_h: optional [hidden_size]
                 )DOC");
}
}    // namespace TEngine
