#include "operator/lstm.hpp"
#include "operator/lstm_param.hpp"
#include "static_graph.hpp"

namespace TEngine {

bool LSTM::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    // input tensors:
    // 0 --- input: [seq_length, batch_size, input_size]
    // 1 --- kernel [ (input_size+hidden_size),cell_state_size]
    // others: optional

    // output tensor: [output_len, batch_size,hidden_size]
    // std::cout<<"!!!!!!!\n";
    const TShape input_shape = ishape[0];
    int batch_size = input_shape.Shape(1);
    if(param_.mxnet_flag == 0)
    {
        batch_size = input_shape.Shape(0);
    }

    //
    std::vector<int> dims(3);
    if(param_.mxnet_flag == 0)
    {
        dims[0] = batch_size;
        dims[1] = input_shape.Shape(0);
        dims[2] = param_.hidden_size;
    }
    else
    {
        dims[0] = input_shape.Shape(0);
        dims[1] = batch_size;
        dims[2] = param_.hidden_size;
    }

    // std::cout<<dims[0]<<","<< dims[1]<<","<<dims[2]<<"\n";

    oshape[0].SetDim(dims);

    return true;
}

void LSTM::SetSchema(void)
{
    Input({"input:float32", "kernel:float32", "bias:float32", "w_f_diag:float32", "w_o_diag:float32",
           "w_i_diag:float32", "project:float32", "init_c:float32", "init_h:float32"})
        .Output({"output:float32"})
        .SetAttr("forget_bias", 0.0f)
        .SetAttr("clip", 0.0f)
        .SetAttr("output_len", 1)
        .SetAttr("sequence_len", 1)
        .SetAttr("input_size", 1)
        .SetAttr("hidden_size", 1)
        .SetAttr("cell_size", 1)
        .SetAttr("has_projection", 0)
        .SetAttr("has_peephole", 0)
        .SetAttr("has_clip", 0)
        .SetAttr("has_bias", 0)
        .SetAttr("has_init_state", 0)
        .SetAttr("forget_act", LSTM_ACT_SIGMOID)
        .SetAttr("input_act", LSTM_ACT_SIGMOID)
        .SetAttr("output_act", LSTM_ACT_SIGMOID)
        .SetAttr("cellin_act", LSTM_ACT_TANH)
        .SetAttr("cellout_act", LSTM_ACT_TANH)
        .SetDoc(R"DOC(LSTM Cell
              input: input sequences, a 3D tensor [seq_length,batch_size,input_size]
              kernel: i/c/f/o weight tensor,[num_directions, 4*hidden_size, ]
              bias:   i/f/c/o bias tensor, [num_directions, 4*hidden_size]
              w_f_diag/w_o_diag/w_i_diag: optional [num_directions, hidden_size]
              init_c/init_h: optional [cell_size]/[hidden_size]
                 )DOC");
}
}    // namespace TEngine
