#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "graph.hpp"
#include "logger.hpp"
#include "node_ops.hpp"
#include "operator/layernormlstm.hpp"
#include "tensor_mem.hpp"
#include "tengine_errno.hpp"
#include <math.h>

namespace TEngine{

namespace LayerNormLSTMRefImpl{

struct LayerNormLSTMOps : public NodeOps
{
    Tensor* input;
    Tensor* i2i_weights_tensor;
    Tensor* i2c_weights_tensor;
    Tensor* i2f_weights_tensor;
    Tensor* i2o_weights_tensor;
    Tensor* igate_bias_tensor;
    Tensor* cgate_bias_tensor;
    Tensor* fgate_bias_tensor;
    Tensor* ogate_bias_tensor;
    Tensor* r2i_weights_tensor;
    Tensor* r2c_weights_tensor;
    Tensor* r2f_weights_tensor;
    Tensor* r2o_weights_tensor;
    Tensor* c2i_weights_tensor;
    Tensor* c2f_weights_tensor;
    Tensor* c2o_weights_tensor;
    Tensor* projection_weights_tensor;
    Tensor* projection_bias_tensor;
    Tensor* iactivationstate_tensor;
    Tensor* icellstate_tensor;
    Tensor* ilayer_norm_coefficients_tensor;
    Tensor* flayer_norm_coefficients_tensor;
    Tensor* clayer_norm_coefficients_tensor;
    Tensor* olayer_norm_coefficients_tensor;
    
    float* input_gate_scratch;
    float* forget_gate_scratch;
    float* cell_scratch;
    float* output_gate_scratch;

    const float LayerNormEpsilon = 1e-8;

    LayerNormLSTMOps(void)
    {
        input = nullptr;
        i2i_weights_tensor = nullptr;
        i2c_weights_tensor = nullptr;
        i2f_weights_tensor = nullptr;
        i2o_weights_tensor = nullptr;
        igate_bias_tensor  = nullptr;
        cgate_bias_tensor  = nullptr;
        fgate_bias_tensor  = nullptr;
        ogate_bias_tensor  = nullptr;
        r2i_weights_tensor = nullptr;
        r2c_weights_tensor = nullptr;
        r2f_weights_tensor = nullptr;
        r2o_weights_tensor = nullptr;
        c2i_weights_tensor = nullptr;
        c2f_weights_tensor = nullptr;
        c2o_weights_tensor = nullptr;
        projection_weights_tensor = nullptr;
        projection_bias_tensor    = nullptr;
        icellstate_tensor         = nullptr;
        iactivationstate_tensor   = nullptr;
        ilayer_norm_coefficients_tensor = nullptr;
        flayer_norm_coefficients_tensor = nullptr;
        clayer_norm_coefficients_tensor = nullptr;
        olayer_norm_coefficients_tensor = nullptr;
        input_gate_scratch  = nullptr;
        forget_gate_scratch = nullptr;
        cell_scratch        = nullptr;
        output_gate_scratch = nullptr;
    }

    void mytanh(float* data, float* output, int size)
    {
        for(int i = 0; i < size; i++)
        {
            output[i] = std::tanh(data[i]);
        }
    }

    void sigmoid(float* data, float* output, int size)
    {
        for(int i = 0; i < size; i++)
        {
            output[i] = 1 / (1 + exp(-data[i]));
        }
    }

    void matbatchvectorwiseproduct(const float* matrix, const float* vector, float* output, int batch, int rows, int cols)
    {
        float* result = output;
        for(int i = 0; i < batch; i++)
        {
            const float* matrix_ptr = matrix;
            for(int j = 0; j < rows; j++)
            {
                double temp = 0.0;
                const float* vector_ptr = vector + i * cols;
                for(int p = 0; p < cols; p++)
                {   
                    temp +=  *matrix_ptr++ * *vector_ptr++;
                }
                //std::cout <<temp <<std::endl;
                *result++ += temp;
            }
        }
    }

    void vectorbatchvectorwiseproductacc(const float* vector1, const float* vector2, float* output, int batch, int rows)
    {
        float* result = output;
        const float* vector_cursor = vector2;
        for(int i = 0; i < batch; i++)
        {
            for(int j = 0; j < rows; j++)
            { 
                *result++ += vector1[j] * *vector_cursor++;
            }
        }
    }

    void vectorbatchvectorwiseproduct(const float* vector1, const float* vector2, float* output, int batch, int rows)
    {
        float* result = output;
        const float* vector_cursor = vector2;
        for(int i = 0; i < batch; i++)
        {
            for(int j = 0; j < rows; j++)
            {
                double temp = 0.0;
                temp =  vector1[j] * *vector_cursor++;
                *result++ = temp;
            }
        }
    }

    void vectorvectorwiseproductacc(const float* vector1, const float* vector2, float*output, int size)
    {
        float* result = output;
        for(int i = 0; i < size; i++)
        {
            double temp = 0.0;
            temp = vector1[i] * vector2[i];
            *result++ += temp;
        }
    }

    void vectorvectorwiseproduct(const float* vector1, const float* vector2, float*output, int size)
    {
        float* result = output;
        for(int i = 0; i < size; i++)
        {
            double temp = 0.0;
            temp = vector1[i] * vector2[i];
            *result++ = temp;
        }
    }

    void vectoradd(const float* vector1, const float* vector2, float* output, int batch, int rows)
    {
        float* result = output;
        for(int i = 0; i < batch; i++)
        {
            for(int j = 0; j < rows; j++)
            {
                *result++ = *vector1++ + vector2[j];
            }
        }

    }

    void vector1sub(const float* input, float* output, int size)
    {
        for(int i = 0; i < size; i++)
        {
            output[i] = 1.0f - input[i];
        }
    }

    void vectorclip(const float* input, float* output, int size, float abs)
    {
        for(int i = 0; i < 0; i++)
        {
            output[i] = std::max(std::min(input[i], abs), -abs);
        }
    }

    void vectorbatchvectorassign(const float* input, float* destvec, int batch, int size)
    {
        for(int i = 0; i < batch; i++)
        {
            memcpy(destvec + i * size, input, size * sizeof(float));
        }
    }

    void layer_norm_nhwc(const float* input, float* output, int cell_size, int batch_size)
    {
        float* result = output;
        for(int i = 0; i < batch_size; i++)
        {
            float sum    = 0.0f;
            float sum_sq = 0.0f;
            for(int j = 0; j < cell_size; j++)
            {
                sum += input[j];
                sum_sq += input[j] * input[j];
            }
            const float mean = sum / cell_size;
            const float variance = sum_sq / cell_size - mean * mean;
            float std_idv = 0;
            if(0 == variance)
            {
                std_idv = 1.0f / std::sqrt(LayerNormEpsilon);
            }
            else
            {
                std_idv = 1.0f / std::sqrt(variance);
            }
            for(int j = 0; j < cell_size; j++)
            {
                *result++ = (input[j] - mean) * std_idv;
            }
            input  += cell_size;
        }
    }   

    void dump_scratch(float* data, int size)
    {
        for(int i = 0; i < size; i++)
        {
            printf("%.7f ",data[i]);
        }
        std::cout << std::endl;
    }

    void activationswitch(float* input, float* output,int batch_size, int cell_size, FusedActivation activationtype)
    {
        switch(activationtype)
        {
            case FusedActivation::kSigmoid:
            {
                sigmoid(input, output, batch_size * cell_size);
                break;
            }
            case FusedActivation::kTanh:
            {
                mytanh(input, output, batch_size * cell_size);
                break;
            }
            default:{
                break;
            }
        }
    }

    void DoLayerNormLSTM(float* input, float* output, 
                         const float* i2i_weights_data, const float* i2c_weights_data, const float* i2f_weights_data, const float* i2o_weights_data, 
                         const float* r2i_weights_data, const float* r2c_weights_data, const float* r2f_weights_data, const float* r2o_weights_data, 
                         const float* c2i_weights_data, const float* c2f_weights_data, const float* c2o_weights_data, 
                         const float* igate_bias_data,  const float* cgate_bias_data,  const float* fgate_bias_data,  const float* ogate_bias_data,
                         const float* projection_bias_data, const float* projection_weights_data,
                         float* icellstate_data, float* iactivationstate_data,
                         const float* ilayer_norm_coefficients_data,
                         const float* flayer_norm_coefficients_data, 
                         const float* clayer_norm_coefficients_data, 
                         const float* olayer_norm_coefficients_data,
                         int batch_size, int cell_size, int input_size, int output_size, int output_real_size,
                         FusedActivation activationtype, float cell_clip, float proj_clip)
    {
        const bool use_cifg = (i2i_weights_data == nullptr);
        const bool use_peephole = (c2o_weights_data != nullptr);
        if(!use_cifg)
        {
            memset(input_gate_scratch, 0, batch_size * cell_size * sizeof(float));
        }
        memset(forget_gate_scratch, 0, batch_size * cell_size * sizeof(float));
        memset(cell_scratch, 0, batch_size * cell_size * sizeof(float));
        memset(output_gate_scratch, 0, batch_size * cell_size * sizeof(float));
        //memset(icellstate_data, 0, batch_size * cell_size * sizeof(float));
        //memset(iactivationstate_data, 0, batch_size * output_size * sizeof(float));
        if(!use_cifg)
        {
            matbatchvectorwiseproduct(i2i_weights_data, input, input_gate_scratch, batch_size, cell_size, input_size);
        }
        //dump_scratch(const_cast<float*>(i2f_weights_data), cell_size*input_size);
        matbatchvectorwiseproduct(i2f_weights_data, input, forget_gate_scratch, batch_size, cell_size, input_size);
        matbatchvectorwiseproduct(i2c_weights_data, input, cell_scratch, batch_size, cell_size, input_size);
        matbatchvectorwiseproduct(i2o_weights_data, input, output_gate_scratch, batch_size, cell_size, input_size);

        //std::cout<<"first deal."<<std::endl;
        //dump_scratch(input_gate_scratch, cell_size*batch_size);
        //dump_scratch(forget_gate_scratch, cell_size*batch_size);
        //dump_scratch(cell_scratch, cell_size*batch_size);
        //dump_scratch(output_gate_scratch, cell_size*batch_size);
        //dump_scratch(icellstate_data, cell_size*batch_size);
        if(!use_cifg)
        {
            matbatchvectorwiseproduct(r2i_weights_data, iactivationstate_data, input_gate_scratch, batch_size, cell_size, output_size);
        }
        matbatchvectorwiseproduct(r2f_weights_data, iactivationstate_data, forget_gate_scratch, batch_size, cell_size, output_size);
        matbatchvectorwiseproduct(r2c_weights_data, iactivationstate_data, cell_scratch, batch_size, cell_size, output_size);
        matbatchvectorwiseproduct(r2o_weights_data, iactivationstate_data, output_gate_scratch, batch_size, cell_size, output_size);
       
        //update input gate
        if(!use_cifg)
        {
            if(use_peephole)
            {
                //dump_scratch(icellstate_data, cell_size*batch_size);
                vectorbatchvectorwiseproductacc(c2i_weights_data, icellstate_data, input_gate_scratch, batch_size, cell_size);
            }
            //dump_scratch(input_gate_scratch, cell_size*batch_size);
            layer_norm_nhwc(input_gate_scratch, input_gate_scratch, cell_size, batch_size);
            //dump_scratch(input_gate_scratch, cell_size*batch_size);
            vectorbatchvectorwiseproduct(ilayer_norm_coefficients_data, input_gate_scratch, input_gate_scratch, batch_size, cell_size);
            //dump_scratch(input_gate_scratch, cell_size*batch_size);
            vectoradd(input_gate_scratch, igate_bias_data, input_gate_scratch, batch_size, cell_size);
            //dump_scratch(input_gate_scratch, cell_size*batch_size);
            sigmoid(input_gate_scratch, input_gate_scratch, cell_size * batch_size);
            //dump_scratch(input_gate_scratch, cell_size*batch_size);
        }
        //std::cout<<"update input gate done."<<std::endl;
        //dump_scratch(input_gate_scratch, cell_size*batch_size);
        //update forget gate
        if(use_peephole)
        {
            vectorbatchvectorwiseproductacc(c2f_weights_data, icellstate_data, forget_gate_scratch, batch_size, cell_size);
        }
        layer_norm_nhwc(forget_gate_scratch, forget_gate_scratch, cell_size, batch_size);
        vectorbatchvectorwiseproduct(flayer_norm_coefficients_data, forget_gate_scratch, forget_gate_scratch, batch_size, cell_size);
        vectoradd(forget_gate_scratch, fgate_bias_data, forget_gate_scratch, batch_size, cell_size);
        sigmoid(forget_gate_scratch, forget_gate_scratch, cell_size * batch_size);
        //std::cout<<"update forget gate."<<std::endl;
        //dump_scratch(forget_gate_scratch, cell_size*batch_size);
        //update cell
        vectorvectorwiseproduct(forget_gate_scratch, icellstate_data, icellstate_data,cell_size * batch_size);
        layer_norm_nhwc(cell_scratch, cell_scratch, cell_size, batch_size);
        vectorbatchvectorwiseproduct(clayer_norm_coefficients_data, cell_scratch, cell_scratch, batch_size, cell_size);
        vectoradd(cell_scratch, cgate_bias_data, cell_scratch, batch_size, cell_size);
        activationswitch(cell_scratch, cell_scratch, batch_size , cell_size, activationtype);
        if(use_cifg)
        {
            //std::cout << "use_cifg."<<std::endl;
            vector1sub(forget_gate_scratch, forget_gate_scratch, batch_size * cell_size);
            vectorvectorwiseproductacc(cell_scratch, forget_gate_scratch, icellstate_data, batch_size * cell_size);
        }
        else
        {
            vectorvectorwiseproductacc(cell_scratch, input_gate_scratch, icellstate_data, batch_size * cell_size);
        }
        //dump_scratch(icellstate_data, cell_size*batch_size);
        if(cell_clip > 0.0)
        {
            vectorclip(cell_scratch, cell_scratch, batch_size * cell_size, cell_clip);
        }
        //std::cout<<"update cell."<<std::endl;
        //dump_scratch(cell_scratch, cell_size*batch_size);
        //update output gate
        //std::cout<<"before update output gate."<<std::endl;
        //dump_scratch(output_gate_scratch, cell_size*batch_size);
        if(use_peephole)
        {
            vectorbatchvectorwiseproductacc(c2o_weights_data, icellstate_data, output_gate_scratch, batch_size, cell_size);
        }
        //std::cout<<"update output gate."<<std::endl;
        //dump_scratch(const_cast<float*>(c2o_weights_data), cell_size);
        layer_norm_nhwc(output_gate_scratch, output_gate_scratch, cell_size, batch_size);   
        vectorbatchvectorwiseproduct(olayer_norm_coefficients_data, output_gate_scratch, output_gate_scratch, batch_size, cell_size);
        vectoradd(output_gate_scratch, ogate_bias_data, output_gate_scratch, batch_size, cell_size);
        sigmoid(output_gate_scratch, output_gate_scratch, batch_size * cell_size);
        activationswitch(icellstate_data, cell_scratch, batch_size, cell_size, activationtype);
        vectorvectorwiseproduct(output_gate_scratch, cell_scratch, output_gate_scratch, batch_size * cell_size);
        const bool use_projection_weight = (projection_weights_data != nullptr);
        const bool use_projection_bias   = (projection_bias_data != nullptr);     

        if(output_real_size == output_size)
        {
            if(use_projection_weight)
            {
                if(use_projection_bias)
                {
                    vectorbatchvectorassign(projection_bias_data, output, batch_size, output_size);
                }
                else
                {
                    memset(output, 0, batch_size * output_size * sizeof(float));
                }
                matbatchvectorwiseproduct(projection_weights_data, output_gate_scratch, output, batch_size, output_size, cell_size);
                if(proj_clip > 0.0)
                {
                    vectorclip(output, output, output_size * batch_size, proj_clip);
                }
                
            }
            else
            {
                memcpy(output, output_gate_scratch, batch_size * output_size * sizeof(float));
            }
            memcpy(iactivationstate_data, output, batch_size * output_size * sizeof(float));
            //dump_scratch(icellstate_data, batch_size * cell_size);
            //dump_scratch(iactivationstate_data, batch_size * output_size);
        }
        else
        {
            if(use_projection_weight)
            {
                if(use_projection_bias)
                {
                    for(int n = 0; n < batch_size; n++)
                    {
                        memcpy(output + n * output_real_size, projection_bias_data, output_real_size);
                    }

                }
                else
                {
                    for(int n = 0; n < batch_size; n++)
                    {
                        memset(output + n * output_real_size, 0, output_real_size * sizeof(float));
                    }
                }
                
                for(int n = 0; n < batch_size; n++)
                {
                    matbatchvectorwiseproduct(projection_weights_data, output_gate_scratch + n * cell_size, 
                                              output + n * output_real_size, 1, output_real_size, cell_size);
                    if(proj_clip > 0.0)
                    {
                        vectorclip(output, output, output_size * batch_size, proj_clip);
                    }
                }
            }
            else
            {
                for(int n = 0; n < batch_size; n++)
                {
                    memcpy(output + n * output_real_size, output_gate_scratch + n * output_size, output_size);
                }
            }
            for(int n = 0; n < batch_size; n++)
            {
                memcpy(iactivationstate_data + n * output_size, output + n * output_real_size, output_size);
            }
            
        }
        
    }

    void getsinglesequence(float* input, float* output, int batch_size,int sequence_size, int per_sequence_size, int sequence_num)
    {
        int per_batch_num = per_sequence_size * sequence_size;
        for(int i = 0; i < batch_size; i++)
        {
            for(int j = 0; j < per_sequence_size; j++)
            {
                output[i * per_sequence_size + j] = input[per_batch_num * i + sequence_num * per_sequence_size + j];
            }
        }
    }
    
    void mixsequence(float* input, float*output, int batch_size, int sequence_size, int per_sequence_size, int sequence_num)
    {
        int per_batch_num = per_sequence_size * sequence_size;
        for(int i = 0; i < batch_size; i++)
        {
            for(int j = 0; j < per_sequence_size; j++)
            {
                output[per_batch_num * i + sequence_num * per_sequence_size + j] = input[i * per_sequence_size + j];    
            }
        }
    }
    
    void dump_tensor_data(Tensor* tensor)
    {
        int size = tensor->GetShape().GetSize();
        float* data = (float*)get_tensor_mem(tensor);
        for(int i = 0; i < size; i++)
        {
            std::cout<<data[i]<<std::endl;
        }
    }

    bool Prerun(Node* node)
    {
        int in_num = node->GetInputNum();

        for(int count = 0; count < in_num; count++)
        {
            Tensor* temptensor = node->GetInputTensor(count);
            const std::string& name = temptensor->GetName();
            if(name.find("input") != std::string::npos)
            {
                input =  temptensor;
            }
            if(name.find("i2i_weights") != std::string::npos)
            {
                i2i_weights_tensor = temptensor;
            }
            if(name.find("i2c_weights") != std::string::npos)
            {
                i2c_weights_tensor = temptensor;
            }
            if(name.find("i2f_weights") != std::string::npos)
            {
                i2f_weights_tensor = temptensor;
            }
            if(name.find("i2o_weights") != std::string::npos)
            {
                i2o_weights_tensor = temptensor;
            }
            if(name.find("r2i_weights") != std::string::npos)
            {
                r2i_weights_tensor = temptensor;
            }
            if(name.find("r2c_weights") != std::string::npos)
            {
                r2c_weights_tensor = temptensor;    
            }
            if(name.find("r2f_weights") != std::string::npos)
            {
                r2f_weights_tensor = temptensor;
            }
            if(name.find("r2o_weights") != std::string::npos)
            {
                r2o_weights_tensor = temptensor;
            }
            if(name.find("c2i_weights") != std::string::npos)
            {
                c2i_weights_tensor = temptensor;    
            }
            if(name.find("c2f_weights") != std::string::npos)
            {
                c2f_weights_tensor = temptensor;    
            }
            if(name.find("c2o_weights") != std::string::npos)
            {
                c2o_weights_tensor = temptensor;        
            }
            if(name.find("igate_bias") != std::string::npos)
            {
                igate_bias_tensor  = temptensor;
            }
            if(name.find("cgate_bias") != std::string::npos)
            {
                cgate_bias_tensor  = temptensor;
            }
            if(name.find("fgate_bias") != std::string::npos)
            {
                fgate_bias_tensor  = temptensor;     
            }
            if(name.find("ogate_bias") != std::string::npos)
            {
                ogate_bias_tensor  = temptensor;    
            }
            if(name.find("projection_weight") != std::string::npos)
            {
                projection_weights_tensor = temptensor;
            }
            if(name.find("projection_bias") != std::string::npos)
            {
                projection_bias_tensor    = temptensor;    
            }
            if(name.find("iactivationstateTensor") != std::string::npos)
            {
                iactivationstate_tensor   = temptensor;
            }
            if(name.find("icellstatetensor") != std::string::npos)
            {
                icellstate_tensor         = temptensor;    
            }
            if(name.find("ilayer_norm_coefficients") != std::string::npos)
            {
                ilayer_norm_coefficients_tensor = temptensor;    
            }
            if(name.find("flayer_norm_coefficients") != std::string::npos)
            {
                flayer_norm_coefficients_tensor = temptensor;
            }
            if(name.find("clayer_norm_coefficients") != std::string::npos)
            {
                clayer_norm_coefficients_tensor = temptensor;
            }
            if(name.find("olayer_norm_coefficients") != std::string::npos)
            {
                olayer_norm_coefficients_tensor = temptensor;
            }
        }
        int batch_size  = input->GetShape().Shape(0);
        int cell_size  = i2o_weights_tensor->GetShape().Shape(0);

        if(!(i2c_weights_tensor==nullptr))
        {
            input_gate_scratch = (float*) malloc(batch_size * cell_size * sizeof(float));
        }
        forget_gate_scratch = (float*) malloc(batch_size * cell_size * sizeof(float));
        cell_scratch = (float*) malloc(batch_size * cell_size * sizeof(float));
        output_gate_scratch = (float*) malloc(batch_size * cell_size * sizeof(float));
        return true;
    }

    bool Run(Node* node)
    {
        LayerNormLSTM* layernormlstm_op = dynamic_cast<LayerNormLSTM*>(node->GetOp());
        LayerNormLSTMParam* param = layernormlstm_op->GetParam();

        Tensor* output = node->GetOutputTensor(0);

        int batch_size  = input->GetShape().Shape(0);
        int input_size = input->GetShape().Shape(input->GetShape().GetDim().size()-1);
        int cell_size  = i2o_weights_tensor->GetShape().Shape(0);
        int output_size = r2o_weights_tensor->GetShape().Shape(1);
        int output_true_size = output->GetShape().Shape(output->GetShape().GetDim().size()-1);
        int sequence_size= input->GetShape().Shape(1);
        const bool use_cifg = (i2i_weights_tensor == nullptr);
        const bool use_peephole = (c2o_weights_tensor != nullptr);


        float* total_input_data = (float*)get_tensor_mem(input);
        float* total_output_data = (float*)get_tensor_mem(output);

        const float* i2i_weights_data = nullptr;
        const float* i2c_weights_data = nullptr;
        const float* i2f_weights_data = nullptr;
        const float* i2o_weights_data = nullptr;
        const float* igate_bias_data  = nullptr;
        const float* cgate_bias_data  = nullptr;
        const float* fgate_bias_data  = nullptr;
        const float* ogate_bias_data  = nullptr;
        const float* r2i_weights_data = nullptr;
        const float* r2c_weights_data = nullptr;
        const float* r2f_weights_data = nullptr;
        const float* r2o_weights_data = nullptr;
        const float* c2i_weights_data = nullptr;
        const float* c2f_weights_data = nullptr;
        const float* c2o_weights_data = nullptr;
        const float* projection_weights_data = nullptr;
        const float* projection_bias_data    = nullptr;
        float* icellstate_data         = nullptr;
        float* iactivationstate_data   = nullptr;
        const float* ilayer_norm_coefficients_data = nullptr;
        const float* flayer_norm_coefficients_data = nullptr;
        const float* clayer_norm_coefficients_data = nullptr;
        const float* olayer_norm_coefficients_data = nullptr;

        if(!use_cifg)
        {
            i2i_weights_data = const_cast<const float*>((float*)get_tensor_mem(i2i_weights_tensor));
            r2i_weights_data = const_cast<const float*>((float*)get_tensor_mem(r2i_weights_tensor));
            igate_bias_data  = const_cast<const float*>((float*)get_tensor_mem(igate_bias_tensor));
        }

        i2c_weights_data = const_cast<const float*>((float*)get_tensor_mem(i2c_weights_tensor));
        i2f_weights_data = const_cast<const float*>((float*)get_tensor_mem(i2f_weights_tensor));
        i2o_weights_data = const_cast<const float*>((float*)get_tensor_mem(i2o_weights_tensor));
        cgate_bias_data  = const_cast<const float*>((float*)get_tensor_mem(cgate_bias_tensor));
        fgate_bias_data  = const_cast<const float*>((float*)get_tensor_mem(fgate_bias_tensor));
        ogate_bias_data  = const_cast<const float*>((float*)get_tensor_mem(ogate_bias_tensor));
        r2c_weights_data = const_cast<const float*>((float*)get_tensor_mem(r2c_weights_tensor));
        r2f_weights_data = const_cast<const float*>((float*)get_tensor_mem(r2f_weights_tensor));
        r2o_weights_data = const_cast<const float*>((float*)get_tensor_mem(r2o_weights_tensor));
        icellstate_data  = (float*)get_tensor_mem(icellstate_tensor);
        iactivationstate_data = (float*)get_tensor_mem(iactivationstate_tensor);

        if(use_peephole)
        {
            if(!use_cifg)
            {
                c2i_weights_data = const_cast<const float*>((float*)get_tensor_mem(c2i_weights_tensor));
            }
            c2f_weights_data = const_cast<const float*>((float*)get_tensor_mem(c2f_weights_tensor));
            c2o_weights_data = const_cast<const float*>((float*)get_tensor_mem(c2o_weights_tensor));
        }

        if(projection_weights_tensor)
        {
            projection_weights_data = const_cast<const float*>((float*)get_tensor_mem(projection_weights_tensor));
        }
        if(projection_bias_tensor)
        {
            projection_bias_data = const_cast<const float*>((float*)get_tensor_mem(projection_bias_tensor));   
        }

        if(!use_cifg)
        {
            ilayer_norm_coefficients_data = const_cast<const float*>((float*)get_tensor_mem(ilayer_norm_coefficients_tensor));
        }
        clayer_norm_coefficients_data = const_cast<const float*>((float*)get_tensor_mem(clayer_norm_coefficients_tensor));
        olayer_norm_coefficients_data = const_cast<const float*>((float*)get_tensor_mem(olayer_norm_coefficients_tensor));
        flayer_norm_coefficients_data = const_cast<const float*>((float*)get_tensor_mem(flayer_norm_coefficients_tensor));
        for(int i = 0; i < sequence_size; i++)
        {
            float* input_data = (float*) malloc(batch_size * input_size * sizeof(float));
            float* output_data = (float*) malloc(batch_size * output_size * sizeof(float));
            memset(output_data, 0, batch_size * output_size * sizeof(float));
            getsinglesequence(total_input_data, input_data, batch_size, sequence_size, input_size, i);
            DoLayerNormLSTM(input_data, output_data, 
                            i2i_weights_data, i2c_weights_data, i2f_weights_data, i2o_weights_data,
                            r2i_weights_data, r2c_weights_data, r2f_weights_data, r2o_weights_data, 
                            c2i_weights_data, c2f_weights_data, c2o_weights_data,
                            igate_bias_data, cgate_bias_data, fgate_bias_data, ogate_bias_data,
                            projection_bias_data, projection_weights_data, 
                            icellstate_data, iactivationstate_data,
                            ilayer_norm_coefficients_data, 
                            flayer_norm_coefficients_data, 
                            clayer_norm_coefficients_data,
                            olayer_norm_coefficients_data,
                            batch_size, cell_size, input_size, output_size, output_true_size,
                            param->fused_activation, param->cell_clip, param->proj_clip);
            mixsequence(output_data, total_output_data, batch_size, sequence_size, output_size, i);
            free(input_data);
            free(output_data);
        }
        return true;
    }

    bool Postrun(Node* node)
    {
        free(input_gate_scratch);
        free(output_gate_scratch);
        free(cell_scratch);
        free(forget_gate_scratch);
        return true;
    }

};
    
}

using namespace LayerNormLSTMRefImpl;

void RegisterLayerNormLSTMNodeExec(void)
{
    LayerNormLSTMOps* ops = new LayerNormLSTMOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "LayerNormLSTM", ops);
} // namespace L


}
