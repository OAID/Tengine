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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: hhchen@openailab.com
 */

#include "../quant_tool.hpp"

//int QuantTool::data_free_quant(const char* model_file, const char* image_dir,
//                    int img_c, int img_h, int img_w, const float* mean, const float* scale,
//                    int num_thread, int sw_RGB, int center_crop)
int QuantTool::data_free_quant()
{
    int letterbox = 0;
    int loop_count = 1;
    const char* image_file = nullptr;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;

    //    /* inital tengine */
    //    if (init_tengine() != 0)
    //    {
    //        fprintf(stderr, "Initial tengine failed.\n");
    //        return -1;
    //    }
    //    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(NULL, "tengine", model_file.c_str());
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return -1;
    }

    struct graph* graphn = (struct graph*)graph;
    // struct node_graph* node_proto = (struct node_graph*)sys_malloc(sizeof(struct node_graph) * graphn->node_num);  // crash access node_proto.input_node_list
    std::vector<node_graph> node_proto(graphn->node_num);

    for (int i = 0; i < graphn->node_num; i++)
    {
        struct node* n = graphn->node_list[i]; //ir node
        const uint16_t node_idx = n->index;    //node idx
        auto op_type = n->op.type;
        const char* layer_name = n->name; //layer name

        const uint16_t input_num = n->input_num;   //input num
        const uint16_t output_num = n->output_num; //output num

        node_proto[i].pass = 0;
        //        node_proto[i].input_node_list = create_vector(sizeof(uint16_t), NULL);
        //        node_proto[i].output_node_list = create_vector(sizeof(uint16_t), NULL);

        for (int j = 0; j < input_num; j++)
        {
            struct tensor* input_tensor = get_ir_graph_tensor(graphn, n->input_tensors[j]);
            const char* input_tensor_name = input_tensor->name;
            uint8_t dim_num = input_tensor->dim_num;

            if (input_tensor->producer >= 0)
            {
                struct node* node = graphn->node_list[input_tensor->producer];
                node_proto[i].input_node_list.push_back(node->index);
                node_proto[node->index].output_node_list.push_back(i);
            }
            if (OP_CONV == op_type || OP_FC == op_type)
            {
                break;
            }
        }
    }

    for (int i = 0; i < graphn->node_num; i++)
    {
        struct node* n = graphn->node_list[i]; //ir node
        const uint16_t node_idx = n->index;    //node idx
        auto op_type = n->op.type;
        const char* layer_name = n->name; //layer name
        if (op_type != NULL)
        {
            if (OP_CONV != op_type && OP_FC != op_type && OP_POOL != op_type)
            {
                if (node_proto[i].input_node_list.size() == 1 && node_proto[i].output_node_list.size() == 1)
                {
                    uint16_t node_input_id = node_proto[i].input_node_list[0];
                    uint16_t node_output_id = node_proto[i].output_node_list[0];
                    if (node_proto[node_input_id].output_node_list.size() == 1 && node_proto[node_output_id].input_node_list.size() == 1)
                    {
                        node_proto[i].input_node_list.erase(node_proto[i].input_node_list.begin() + 0);
                        node_proto[i].output_node_list.erase(node_proto[i].output_node_list.begin() + 0);

                        node_proto[node_input_id].output_node_list.erase(node_proto[node_input_id].output_node_list.begin() + 0);
                        node_proto[node_input_id].output_node_list.push_back(node_output_id);

                        node_proto[node_output_id].input_node_list.erase(node_proto[node_output_id].input_node_list.begin() + 0);
                        node_proto[node_output_id].input_node_list.push_back(node_input_id);
                    }
                }
            }
        }
    }

    for (int i = 0; i < graphn->node_num; i++)
    {
        struct node* n = graphn->node_list[i]; //ir node
        const uint16_t node_idx = n->index;    //node idx
        op_name = n->op.type;
        const char* layer_name = n->name; //layer name

        const uint16_t input_num = n->input_num;   //input num
        const uint16_t output_num = n->output_num; //output num

        if (op_name != NULL)
        {
            if (OP_CONV == op_name)
            {
                // DW_Conv && Direct_Conv
                struct conv_param* conv_param = (struct conv_param*)n->op.param_mem;
                if (conv_param->group == conv_param->output_channel && conv_param->group != 1)
                {
                    //                    printf("    #### DW Conv ####\n");
                    if (node_proto[i].input_node_list.size() == 1 && node_proto[i].output_node_list.size() == 1)
                    {
                        uint16_t node_input_id = node_proto[i].input_node_list[0];
                        uint16_t node_output_id = node_proto[i].output_node_list[0];
                        auto op_name0 = graphn->node_list[node_input_id]->op.type;
                        auto op_name2 = graphn->node_list[node_output_id]->op.type;

                        if (node_proto[node_input_id].output_node_list.size() == 1 && node_proto[node_output_id].input_node_list.size() == 1 && OP_CONV == op_name0 && OP_CONV == op_name2)
                        {
                            node_proto[i].pass = 1;              //layer1
                            node_proto[node_input_id].pass = 1;  //layer0
                            node_proto[node_output_id].pass = 1; //layer2

                            // layer0 min/max range
                            struct node* nodeP = graphn->node_list[node_input_id];
                            struct tensor* input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[1]);
                            uint16_t dims0 = input_tensor->dims[0];
                            uint16_t dims123 = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];

                            std::vector<float> layer0_max(dims0, 0.0f);
                            std::vector<float> layer0_min(dims0, 0.0f);
                            std::vector<float> layer0_range(dims0, 0.0f);

                            float* data_layer0 = (float*)input_tensor->data;
                            for (int d0 = 0; d0 < dims0; d0++)
                            {
                                for (int d1 = 0; d1 < dims123; d1++)
                                {
                                    if (data_layer0[dims123 * d0 + d1] > layer0_max[d0])
                                        layer0_max[d0] = data_layer0[dims123 * d0 + d1];
                                    if (data_layer0[dims123 * d0 + d1] < layer0_max[d0])
                                        layer0_min[d0] = data_layer0[dims123 * d0 + d1];
                                }
                            }
                            //                            printf("### %d ###\n",dims0);
                            for (int d0 = 0; d0 < dims0; d0++)
                            {
                                layer0_range[d0] = layer0_max[d0] - layer0_min[d0];
                            }

                            // layer1 min/max range
                            nodeP = graphn->node_list[i];
                            input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[1]);
                            dims0 = input_tensor->dims[0];
                            dims123 = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];

                            std::vector<float> layer1_max(dims0, 0.0f);
                            std::vector<float> layer1_min(dims0, 0.0f);
                            std::vector<float> layer1_range(dims0, 0.0f);

                            float* data_layer1 = (float*)input_tensor->data;
                            for (int d0 = 0; d0 < dims0; d0++)
                            {
                                for (int d1 = 0; d1 < dims123; d1++)
                                {
                                    if (data_layer1[dims123 * d0 + d1] > layer1_max[d0])
                                        layer1_max[d0] = data_layer1[dims123 * d0 + d1];
                                    if (data_layer1[dims123 * d0 + d1] < layer1_max[d0])
                                        layer1_min[d0] = data_layer1[dims123 * d0 + d1];
                                }
                            }
                            //                            printf("### %d ###\n",dims0);
                            for (int d0 = 0; d0 < dims0; d0++)
                            {
                                layer1_range[d0] = layer1_max[d0] - layer1_min[d0];
                            }

                            // layer2 min/max range
                            nodeP = graphn->node_list[node_output_id];
                            input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[1]);
                            dims0 = input_tensor->dims[0];
                            uint16_t dims1 = input_tensor->dims[1];
                            uint16_t dims23 = input_tensor->dims[2] * input_tensor->dims[3];

                            std::vector<float> layer2_max(dims0, 0.0f);
                            std::vector<float> layer2_min(dims0, 0.0f);
                            std::vector<float> layer2_range(dims0, 0.0f);

                            float* data_layer2 = (float*)input_tensor->data;
                            for (int d0 = 0; d0 < dims0; d0++)
                            {
                                for (int d1 = 0; d1 < dims1; d1++)
                                {
                                    for (int d2 = 0; d2 < dims23; d2++)
                                    {
                                        if (data_layer2[dims1 * dims23 * d0 + dims23 * d1 + d2] > layer2_max[d1])
                                        {
                                            layer2_max[d1] = data_layer2[dims1 * dims23 * d0 + dims23 * d1 + d2];
                                        }
                                        if (data_layer2[dims1 * dims23 * d0 + dims23 * d1 + d2] < layer2_min[d1])
                                        {
                                            layer2_min[d1] = data_layer2[dims1 * dims23 * d0 + dims23 * d1 + d2];
                                        }
                                    }
                                }
                            }
                            //                            printf("### %d ###\n",dims1);
                            for (int d1 = 0; d1 < dims1; d1++)
                            {
                                layer2_range[d1] = layer2_max[d1] - layer2_min[d1];
                            }

                            //////////////////////////////////////////////////////////////////////////////////

                            // layer ops sqrt
                            // float ops_range[dims1];  // dims1 should be constant
                            float* ops_range = new float[dims1];
                            for (int ops = 0; ops < dims1; ops++)
                            {
                                ops_range[ops] = pow(layer0_range[ops] * layer1_range[ops] * layer2_range[ops], 1.0 / 3);
                            }

                            // float S01[dims1];
                            // float S01_F[dims1];
                            // float S12[dims1];
                            // float S12_F[dims1];
                            float* S01 = new float[dims1];
                            float* S01_F = new float[dims1];
                            float* S12 = new float[dims1];
                            float* S12_F = new float[dims1];

                            for (int ops = 0; ops < dims1; ops++)
                            {
                                if (ops_range[ops] == 0)
                                {
                                    S01[ops] = 0.0;
                                    S12_F[ops] = 0.0;
                                }
                                else
                                {
                                    S01[ops] = layer0_range[ops] / ops_range[ops];
                                    S12_F[ops] = layer2_range[ops] / ops_range[ops];
                                }
                                if (layer0_range[ops] == 0)
                                    S01_F[ops] = 0.0;
                                else
                                    S01_F[ops] = ops_range[ops] / layer0_range[ops];
                                if (layer2_range[ops] == 0)
                                    S12[ops] = 0.0;
                                else
                                    S12[ops] = ops_range[ops] / layer2_range[ops];
                            }
                            //////////////////////////////////////////////////////////////////////////////////

                            // layer0 output
                            nodeP = graphn->node_list[node_input_id];
                            input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[1]);
                            dims0 = input_tensor->dims[0];
                            dims123 = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];
                            for (int d0 = 0; d0 < dims0; d0++)
                            {
                                for (int d1 = 0; d1 < dims123; d1++)
                                {
                                    data_layer0[dims123 * d0 + d1] = data_layer0[dims123 * d0 + d1] * S01_F[d0];
                                }
                            }
                            input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[2]);
                            dims0 = input_tensor->dims[0];
                            float* data_layer0_bias = (float*)sys_malloc(sizeof(float) * dims0);
                            data_layer0_bias = (float*)input_tensor->data;
                            for (int d0 = 0; d0 < dims0; d0++)
                            {
                                data_layer0_bias[d0] = data_layer0_bias[d0] * S01_F[d0];
                            }

                            // layer1 output
                            nodeP = graphn->node_list[i];
                            input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[1]);
                            dims0 = input_tensor->dims[0];
                            dims123 = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];
                            for (int d0 = 0; d0 < dims0; d0++)
                            {
                                for (int d1 = 0; d1 < dims123; d1++)
                                {
                                    data_layer1[dims123 * d0 + d1] = data_layer1[dims123 * d0 + d1] * S01[d0] * S12_F[d0];
                                }
                            }
                            input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[2]);
                            dims0 = input_tensor->dims[0];
                            float* data_layer1_bias = (float*)input_tensor->data;
                            for (int d0 = 0; d0 < dims0; d0++)
                            {
                                data_layer1_bias[d0] = data_layer1_bias[d0] * S12_F[d0];
                            }

                            // layer2 output
                            nodeP = graphn->node_list[node_output_id];
                            input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[1]);
                            dims0 = input_tensor->dims[0];
                            dims1 = input_tensor->dims[1];
                            dims23 = input_tensor->dims[2] * input_tensor->dims[3];
                            for (int d0 = 0; d0 < dims0; d0++)
                            {
                                for (int d1 = 0; d1 < dims1; d1++)
                                {
                                    for (int d2 = 0; d2 < dims23; d2++)
                                    {
                                        data_layer2[dims1 * dims23 * d0 + dims23 * d1 + d2] = data_layer2[dims1 * dims23 * d0 + dims23 * d1 + d2] * S12[d1];
                                    }
                                }
                            }
                            delete[] S01; // free the memory
                            S01 = NULL;
                            delete[] S01_F;
                            S01_F = NULL;
                            delete[] S12;
                            S12 = NULL;
                            delete[] S12_F;
                            S12_F = NULL;
                            delete[] ops_range;
                            ops_range = NULL;
                        }
                    }
                }
                else
                {
                    //                    printf("    #### Direct Conv ####\n");
                    if (node_proto[i].pass == 0)
                    {
                        if (node_proto[i].input_node_list.size() == 1)
                        {
                            uint16_t node_input_id = node_proto[i].input_node_list[0];
                            if (graphn->node_list[node_input_id]->input_num > 0)
                            {
                                auto op_name0 = graphn->node_list[node_input_id]->op.type;

                                if (node_proto[node_input_id].output_node_list.size() == 1 && op_name0 == OP_CONV)
                                {
                                    struct conv_param* conv_param0 = (struct conv_param*)graphn->node_list[node_input_id]->op.param_mem;
                                    if (conv_param0->group != conv_param0->output_channel || conv_param0->group == 1)
                                    {
                                        node_proto[i].pass = 1;             //layer1
                                        node_proto[node_input_id].pass = 1; //layer0

                                        // layer0 min/max range
                                        struct node* nodeP = graphn->node_list[node_input_id];
                                        struct tensor* input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[1]);
                                        uint16_t dims0 = input_tensor->dims[0];
                                        uint16_t dims123 = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];

                                        std::vector<float> layer0_max(dims0, 0.0f);
                                        std::vector<float> layer0_min(dims0, 0.0f);
                                        std::vector<float> layer0_range(dims0, 0.0f);

                                        float* data_layer0 = (float*)input_tensor->data;
                                        for (int d0 = 0; d0 < dims0; d0++)
                                        {
                                            for (int d1 = 0; d1 < dims123; d1++)
                                            {
                                                if (data_layer0[dims123 * d0 + d1] >= layer0_max[d0])
                                                    layer0_max[d0] = data_layer0[dims123 * d0 + d1];
                                                if (data_layer0[dims123 * d0 + d1] < layer0_max[d0])
                                                    layer0_min[d0] = data_layer0[dims123 * d0 + d1];
                                            }
                                        }
                                        //                                    printf("### %d ###\n",dims0);
                                        for (int d0 = 0; d0 < dims0; d0++)
                                        {
                                            layer0_range[d0] = layer0_max[d0] - layer0_min[d0];
                                        }

                                        // layer1 min/max range
                                        nodeP = graphn->node_list[i];
                                        input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[1]);
                                        dims0 = input_tensor->dims[0];
                                        uint16_t dims1 = input_tensor->dims[1];
                                        uint16_t dims23 = input_tensor->dims[2] * input_tensor->dims[3];

                                        std::vector<float> layer1_max(dims1, 0.0f);
                                        std::vector<float> layer1_min(dims1, 0.0f);
                                        std::vector<float> layer1_range(dims1, 0.0f);

                                        float* data_layer1 = (float*)input_tensor->data;
                                        for (int d0 = 0; d0 < dims0; d0++)
                                        {
                                            for (int d1 = 0; d1 < dims1; d1++)
                                            {
                                                for (int d2 = 0; d2 < dims23; d2++)
                                                {
                                                    if (data_layer1[dims1 * dims23 * d0 + dims23 * d1 + d2] >= layer1_max[d1])
                                                    {
                                                        layer1_max[d1] = data_layer1[dims1 * dims23 * d0 + dims23 * d1 + d2];
                                                    }
                                                    if (data_layer1[dims1 * dims23 * d0 + dims23 * d1 + d2] < layer1_min[d1])
                                                    {
                                                        layer1_min[d1] = data_layer1[dims1 * dims23 * d0 + dims23 * d1 + d2];
                                                    }
                                                }
                                            }
                                        }
                                        //                                    printf("### %d ###\n",dims1);
                                        for (int d0 = 0; d0 < dims1; d0++)
                                        {
                                            layer1_range[d0] = layer1_max[d0] - layer1_min[d0];
                                        }

                                        //////////////////////////////////////////////////////////////////////////////////

                                        // layer ops sqrt
                                        // float ops_range[dims1];   // dims1 should be constant
                                        float* ops_range = new float[dims1];
                                        for (int ops = 0; ops < dims1; ops++)
                                        {
                                            ops_range[ops] = sqrt(layer0_range[ops] * layer1_range[ops]);
                                        }

                                        // float S01[dims1];
                                        // float S01_F[dims1];
                                        float* S01 = new float[dims1];
                                        float* S01_F = new float[dims1];

                                        for (int ops = 0; ops < dims1; ops++)
                                        {
                                            if (ops_range[ops] == 0)
                                            {
                                                S01[ops] = 0.0;
                                            }
                                            else
                                            {
                                                S01[ops] = layer0_range[ops] / ops_range[ops];
                                            }
                                            if (layer0_range[ops] == 0)
                                                S01_F[ops] = 0.0;
                                            else
                                                S01_F[ops] = ops_range[ops] / layer0_range[ops];
                                        }
                                        //////////////////////////////////////////////////////////////////////////////////
                                        // layer0 output
                                        nodeP = graphn->node_list[node_input_id];
                                        input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[1]);
                                        dims0 = input_tensor->dims[0];
                                        dims123 = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];
                                        for (int d0 = 0; d0 < dims0; d0++)
                                        {
                                            for (int d1 = 0; d1 < dims123; d1++)
                                            {
                                                data_layer0[dims123 * d0 + d1] = data_layer0[dims123 * d0 + d1] * S01_F[d0];
                                            }
                                        }
                                        input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[2]);
                                        dims0 = input_tensor->dims[0];
                                        float* data_layer0_bias = (float*)sys_malloc(sizeof(float) * dims0);
                                        data_layer0_bias = (float*)input_tensor->data;
                                        for (int d0 = 0; d0 < dims0; d0++)
                                        {
                                            data_layer0_bias[d0] = data_layer0_bias[d0] * S01_F[d0];
                                        }

                                        // layer1 output
                                        nodeP = graphn->node_list[i];
                                        input_tensor = get_ir_graph_tensor(graphn, nodeP->input_tensors[1]);
                                        dims0 = input_tensor->dims[0];
                                        dims1 = input_tensor->dims[1];
                                        dims23 = input_tensor->dims[2] * input_tensor->dims[3];
                                        for (int d0 = 0; d0 < dims0; d0++)
                                        {
                                            for (int d1 = 0; d1 < dims1; d1++)
                                            {
                                                for (int d2 = 0; d2 < dims23; d2++)
                                                {
                                                    data_layer1[dims1 * dims23 * d0 + dims23 * d1 + d2] = data_layer1[dims1 * dims23 * d0 + dims23 * d1 + d2] * S01[d1];
                                                }
                                            }
                                        }
                                        delete[] S01; // free the memory
                                        S01 = NULL;
                                        delete[] S01_F;
                                        S01_F = NULL;
                                        delete[] ops_range;
                                        ops_range = NULL;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (!save_graph(graph, "test_dfq_fp32.tmfile"))
    {
        fprintf(stderr, "save graph failed.\n");
        return -1;
    }

    /* set the shape, data buffer of input_tensor of the graph */
    int img_size = img_h * img_w * img_c;
    int dims[] = {1, img_c, img_h, img_w}; // nchw
    float* input_data = (float*)malloc(img_size * sizeof(float));

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    std::vector<std::string> imgs_list;
    if (image_dir.c_str() != NULL)
    {
        readFileList(image_dir, imgs_list);
    }
    else
    {
        imgs_list.push_back(image_file);
    }
    uint32_t img_num = imgs_list.size();

    /* prepare process input data, set the data mem to input tensor */
    get_input_data_cv(imgs_list[0].c_str(), input_data, img_c, img_h, img_w, mean, scale,
                      1, 0, 0, 0, 0);

    /* run graph */
    for (int i = 0; i < loop_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
    }

    /* get the result of classification */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* output_data = (float*)get_tensor_buffer(output_tensor);
    int output_size = get_tensor_buffer_size(output_tensor) / sizeof(float);

    //    printf("out put data %f %d \n",output_data[0], output_size);
    fprintf(stderr, "--------------------------------------\n");

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    //    release_tengine();

    return 0;
}
