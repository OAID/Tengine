/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, Open AI Lab
 * Author: jjzeng@openailab.com
 */

#include "tengine_fullyconnected_op.hpp"
#include "tengine_cpp_api.h"

#include <memory.h>

namespace tengine
{
    namespace nn
    {

        int create_input_node(graph_t graph, const char* node_name, int batch,int inch, int in_h, int in_w)
        {
            node_t node     = create_graph_node(graph, node_name, "InputOp");
            tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
            set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

            int dims[4] = {batch, inch, in_h, in_w};
            set_tensor_shape(tensor, dims, 4);

            release_graph_tensor(tensor);
            release_graph_node(node);

            return 0;
        }


        int create_fc_graph(graph_t graph,Tensor& input,const char* weight,const char* bias,int num_output)
        {
            const char* data_name = "data";
            const char* output_name = "data/output";
            const char* node_name = "fc";
            const char* weight_name = "data/weight";
            const char* bias_name = "data/bias";

            if (create_input_node(graph, data_name, input.n,input.c, input.h, input.w) != 0)
            {
                return -1;
            }

            node_t fc_node = create_graph_node(graph,node_name,"FullyConnected");
            set_node_attr_int(fc_node, "num_output", &num_output);

            /* set input data */
            tensor_t input_tensor = get_graph_tensor(graph, data_name);
            set_node_input_tensor(fc_node, 0, input_tensor);
            //set_tensor_buffer(input_tensor, input.data_, input.n_ * input.c_ * input.h_ * input.w_ * sizeof(float));
            release_graph_tensor(input_tensor);

            /* output */
            tensor_t output_tensor = create_graph_tensor(graph, output_name, TENGINE_DT_FP32);
            set_node_output_tensor(fc_node, 0, output_tensor, TENSOR_TYPE_VAR);

            /* weight */
            int w_dims[] = {num_output, input.c * input.h * input.w};
            node_t w_node = create_graph_node(graph, weight_name, "Const");
            tensor_t w_tensor = create_graph_tensor(graph, weight_name, TENGINE_DT_FP32);
            set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
            set_node_input_tensor(fc_node, 1, w_tensor);
            set_tensor_shape(w_tensor, w_dims, 2);

            /* set weight data */
            int buf_size = get_tensor_buffer_size(w_tensor);
            
            set_tensor_buffer(w_tensor, (void*)weight, buf_size);
            release_graph_node(w_node);
            release_graph_tensor(w_tensor);

            /* bias */
            if(bias != NULL)
            {
                int b_dims[] = {num_output};
                node_t b_node = create_graph_node(graph, bias_name, "Const");
                tensor_t b_tensor = create_graph_tensor(graph, bias_name, TENGINE_DT_FP32);
                set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);
                set_tensor_shape(b_tensor, b_dims, 1);
                set_node_input_tensor(fc_node, 2, b_tensor);
                /* set bias data */
                int buf_size = get_tensor_buffer_size(b_tensor);
                set_tensor_buffer(b_tensor, (void*)bias, buf_size);

                release_graph_node(b_node);
                release_graph_tensor(b_tensor);
            }

            /* set input/output node */
            const char* inputs_name[]  = {data_name};
            const char* outputs_name[] = {node_name};
            if (set_graph_input_node(graph, inputs_name, sizeof(inputs_name) / sizeof(char*)) < 0)
            {
                return -1;
            }

            if (set_graph_output_node(graph, outputs_name, sizeof(outputs_name) / sizeof(char*)) < 0)
            {
                return -1;
            }

            return 0;
        }

        bool TengineFullyConnected::init(Tensor& input,const char* weight,const char* bias,int num_output)
        {
            init_tengine();
            _graph = create_graph(NULL, NULL, NULL);
            if(_graph == NULL)
            {
                return false;
            }

            if( 0 != create_fc_graph(_graph,input,weight,bias,num_output) )
            {
                destroy_graph(_graph);
                return false;
            }

            return true;
        }

        TTengineOpPtr TengineFullyConnected::create(Tensor& input,const char* weight,const char* bias,int num_output)
        {
            TengineFullyConnected* op = new TengineFullyConnected();
            op->init(input,weight,bias,num_output);
            return TTengineOpPtr(op);
        }

        bool TengineFullyConnected::run()
        {
            if( _graph == NULL )
            {
                return false;
            }

            if( prerun_graph(_graph) < 0 )
            {
                return false;
            }

            if(run_graph(_graph, 1) < 0)
            {
                return false;
            }

            return true;
        }

        Tensor* TengineFullyConnected::get_output_tensor()const
        {
            node_t out_node = get_graph_output_node(_graph,0);
            tensor_t out_tensor = get_node_output_tensor(out_node,0);

            int dims[4] = {0};
            int dim_num = 4;
            dim_num = get_tensor_shape(out_tensor, dims, dim_num);
            if(dim_num < 0)
            {
                std::printf("Get tensor shape failed\n");
                return NULL;
            }

            Tensor* t = NULL;
            if(dim_num == 4)
            {
                t = new Tensor(dims[0],dims[3], dims[2], dims[1], 4);
            }
            else
            {
                return NULL;
            }

            int buffer_size = get_tensor_buffer_size(out_tensor);
            void* buffer = (get_tensor_buffer(out_tensor));

            memcpy(t->data, buffer, buffer_size);

            return t;

        }

        TengineFullyConnected::~TengineFullyConnected()
        {
            if( _graph )
            {
                postrun_graph(_graph);
                destroy_graph(_graph);
                _graph = NULL;
            }
        }

        bool TengineFullyConnected::valid()const
        {
            return _graph != NULL;
        }

    }

}
