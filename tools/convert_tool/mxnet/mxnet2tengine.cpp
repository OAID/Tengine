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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: xlchen@openailab.com
 */

#include "mxnet2tengine.hpp"

typedef std::string::size_type pos;
const int OP_VERSION = 1;
typedef std::map<std::string, std::string>::const_iterator const_iterator;

std::vector<int>& split(const std::string& str, char delim, std::vector<int>& elems, bool skip_empty = true)
{
    std::istringstream iss(str);
    for (std::string item; getline(iss, item, delim);)
        if (skip_empty && item.empty())
            continue;
        else
            elems.push_back(atoi(item.c_str()));
    return elems;
}

static void ParseAttr_n(const std::string str, std::vector<int>& result)
{
    std::string s = str.substr(1, str.length() - 2);
    split(s, ',', result);
}

// Parse "(xxx, yyy)"
static void ParseAttr(const std::string str, std::vector<int>& result)
{
    // Remove leading '(' and trailing ')'
    std::string s = str.substr(1, str.length() - 2);

    std::string s1, s2;
    int i;
    while (1)
    {
        pos comma_pos = s.find(',');
        if (comma_pos != std::string::npos)
        {
            s1 = s.substr(0, comma_pos);
            s2 = s.substr(comma_pos + 1);
            s2.erase(0, s2.find_first_not_of(" "));
            std::istringstream ist1(s1);
            ist1 >> i;
            result.push_back(i);
            s = s2;
        }
        else
        {
            std::istringstream ist2(s2);
            ist2 >> i;
            result.push_back(i);
            break;
        }
    }
}

static int change_node_op(ir_node_t* node, int new_op_type)
{
    sys_free(node->op.param_mem);
    node->op.type = new_op_type;
    ir_method_t* ir_method = find_op_method(new_op_type, OP_VERSION);
    if ((NULL != ir_method) && (NULL != ir_method->init) && (ir_method->init(&node->op) < 0))
    {
        return -1;
    }

    return 0;
}

bool mxnet_serializer::find_op_load_method(const std::string& op_name)
{
    if (op_load_map.count(op_name))
        return true;

    return false;
}

int mxnet_serializer::load_graph_node(ir_graph_t* graph, std::vector<MxnetNode>& nodelist, std::vector<MxnetParam>& paramlist)
{
    unsigned int i;
    std::vector<std::string> no_supported_op;
    for (i = 0; i < nodelist.size(); i++)
    {
        MxnetNode mxnet_node = nodelist.at(i);
        std::string mxnet_op_name = mxnet_node.op;
        if (mxnet_node.op == "null" || mxnet_node.op == "_zeros")
            continue;
        if (!find_op_load_method(mxnet_node.op))
        {
            auto it = find(no_supported_op.begin(), no_supported_op.end(), mxnet_op_name);
            if (it == no_supported_op.end())
                no_supported_op.push_back(mxnet_node.op);
        }
    }
    if (no_supported_op.size())
    {
        fprintf(stderr, "These %zu op are not supported\n{ ", no_supported_op.size());
        for (int j = 0; j < (int)no_supported_op.size(); j++)
        {
            fprintf(stderr, "%s ", no_supported_op[j].c_str());
        }
        fprintf(stderr, "}\n");
        return -1;
    }

    for (i = 0; i < nodelist.size(); i++)
    {
        MxnetNode mxnet_node = nodelist.at(i);
        std::string op_name = mxnet_node.op;
        std::string node_name = mxnet_node.name;

        if (op_name == "null" || op_name == "_zeros")
            continue;

        ir_node_t* ir_node = create_ir_node(graph, node_name.c_str(), op_load_map[op_name].first, OP_VERSION);
        if (ir_node == NULL)
        {
            return -1;
        }

        // set node io
        int input_number = mxnet_node.inputs.size();
        for (int j = 0; j < input_number; j++)
        {
            int input_idx = mxnet_node.inputs.at(j);
            const MxnetNode& input_node = nodelist.at(input_idx);

            if (input_node.name.find("label") != std::string::npos || input_node.name.find("state") != std::string::npos)
                continue;

            int tensor_id = get_ir_tensor_index_from_name(graph, input_node.name.c_str());
            ir_tensor_t* tensor = get_ir_graph_tensor(graph, tensor_id);
            set_ir_node_input_tensor(ir_node, j, tensor);
        }

        const std::string& output_name = mxnet_node.name;
        ir_tensor_t* output_tensor = create_ir_tensor(graph, output_name.c_str(), TENGINE_DT_FP32);
        set_ir_node_output_tensor(ir_node, 0, output_tensor);

        op_load_t loader = op_load_map[op_name].second;
        if (loader(graph, ir_node, mxnet_node) < 0)
        {
            fprintf(stderr, "load op %s func failed in node %s .\n", op_name.c_str(), node_name.c_str());
            return -1;
        }
    }

    if (i < nodelist.size())
        return -1;

    return 0;
}

static ir_tensor_t* find_tensor(ir_graph_t* graph, const std::string& tensor_name)
{
    for (uint16_t i = 0; i < graph->tensor_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        if (tensor->name == tensor_name)
        {
            return tensor;
        }
    }

    return nullptr;
}

static bool get_param(const std::string name, const std::vector<MxnetParam>& paramlist, MxnetParam& param)
{
    for (unsigned int i = 0; i < paramlist.size(); i++)
    {
        if (paramlist.at(i).name == name)
        {
            param = paramlist.at(i);
            return true;
        }
    }
    return false;
}

int mxnet_serializer::set_graph_input(ir_graph_t* graph, std::vector<MxnetNode>& nodelist, std::vector<MxnetParam>& paramlist)
{
    std::vector<int16_t> input_nodes;
    for (unsigned int i = 0; i < nodelist.size(); i++)
    {
        const MxnetNode& mxnet_node = nodelist.at(i);
        if (mxnet_node.name == "data" || mxnet_node.name == "x" || mxnet_node.name == "y")
        {
            if (find_tensor(graph, mxnet_node.name) != nullptr)
                continue;

            ir_tensor_t* tensor = create_ir_tensor(graph, mxnet_node.name.c_str(), TENGINE_DT_FP32);
            if (tensor == NULL)
            {
                fprintf(stderr, "create input tensor failed.\n");
                return -1;
            }
            MxnetParam param;
            if (get_param(mxnet_node.name, paramlist, param))
            {
                set_ir_tensor_shape(tensor, param.dims.data(), param.dim_size);
            }

            tensor->tensor_type = TENSOR_TYPE_INPUT;
            ir_node_t* node = create_ir_node(graph, tensor->name, OP_INPUT, OP_VERSION);
            if (node == NULL)
            {
                fprintf(stderr, "create input node failed.\n");
                return -1;
            }
            set_ir_node_output_tensor(node, 0, tensor);
            input_nodes.push_back(node->index);
        }
    }
    set_ir_graph_input_node(graph, input_nodes.data(), input_nodes.size());
    return 0;
}

int mxnet_serializer::set_graph_output(ir_graph_t* graph)
{
    std::vector<int16_t> output_nodes;
    for (int i = 0; i < graph->tensor_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        if (tensor->tensor_type == TENSOR_TYPE_VAR && tensor->consumer_num == 0)
        {
            output_nodes.push_back(tensor->producer);
        }
    }

    set_ir_graph_output_node(graph, output_nodes.data(), output_nodes.size());
    return 0;
}

int mxnet_serializer::load_constant_tensor(ir_graph_t* graph, std::vector<MxnetNode>& nodelist, std::vector<MxnetParam>& paramlist)
{
    int const_tensor_number = paramlist.size();
    std::set<std::string> node_name_set;

    for (unsigned int i = 0; i < nodelist.size(); i++)
    {
        const MxnetNode& mxnet_node = nodelist.at(i);
        node_name_set.insert(mxnet_node.name);
    }

    for (int i = 0; i < const_tensor_number; i++)
    {
        const MxnetParam& mxnet_tensor = paramlist.at(i);

        if (!node_name_set.count(mxnet_tensor.name))
        {
            fprintf(stderr, "skip tensor:%s.\n", mxnet_tensor.name.c_str());
            continue;
        }

        ir_tensor_t* tensor = create_ir_tensor(graph, mxnet_tensor.name.c_str(), TENGINE_DT_FP32);
        if (tensor == NULL)
        {
            fprintf(stderr, "create ir tensor failed!\n");
            return -1;
        }

        set_ir_tensor_shape(tensor, mxnet_tensor.dims.data(), mxnet_tensor.dim_size);
        tensor->tensor_type = TENSOR_TYPE_CONST;
        int tensor_size = tensor->elem_num * sizeof(float);
        tensor->data = sys_malloc(tensor_size);
        float* mem_buf = (float*)tensor->data;
        float* raw_data = (float*)mxnet_tensor.raw_data;
        for (int j = 0; j < tensor->elem_num; j++)
        {
            mem_buf[j] = raw_data[j];
        }

        /* Now, create the node .... */
        ir_node_t* node = create_ir_node(graph, tensor->name, OP_CONST, OP_VERSION);
        set_ir_node_output_tensor(node, 0, tensor);
    }

    return 0;
}

static void Trim(std::string& s, const char charlist[])
{
    // Erase the leading characters
    s.erase(0, s.find_first_not_of(charlist));
    // Erase the trailing characters
    s.erase(s.find_last_not_of(charlist) + 1);
}

static bool ParseNodeParam(const std::string& str, std::string& param, std::string& value)
{
    pos colon_pos = str.find(':');
    if (colon_pos != std::string::npos)
    {
        param = str.substr(0, colon_pos);
        Trim(param, " \t\f\v\n\r\",");

        value = str.substr(colon_pos + 1);
        Trim(value, " \t\f\v\n\r\",");

        return true;
    }
    return -1;
}

static void ParseInputList(const std::string& str, std::vector<int>& inputs)
{
    // Remove leading '[' and trailing ']'
    std::string s = str.substr(1, str.length() - 2);

    if (s.empty())
        return;

    pos bracket_pos = s.find('[');
    while (bracket_pos != std::string::npos)
    {
        int i;
        s = s.substr(bracket_pos + 1);
        std::istringstream ist(s.substr(0, s.find(',')));
        ist >> i;
        inputs.push_back(i);
        bracket_pos = s.find('[');
    }

#ifdef DEBUG
    std::cout << "Parse Input List " << str << " : ";
    for (unsigned int i = 0; i < inputs.size(); i++)
        std::cout << inputs.at(i) << " ";
    std::cout << std::endl;
#endif
}

int mxnet_serializer::load_text_file(std::string model_file, std::vector<MxnetNode>& nodelist)
{
    unsigned int nest = 0; /* 0 : out of define body
                              1 : in the define body
                              2 : in the nodes list
                              3 : in the node block
                              4 : in the attr block */

    enum
    {
        OUT_DEF_BODY,
        IN_DEF_BODY,
        IN_NODES_LIST,
        IN_NODE_BLOCK,
        IN_ATTR_BLOCK
    };

    static const char start_def_body[] = "{";
    static const char start_nodes_list[] = "\"nodes\": [";
    static const char end_nodes_list[] = "],";
    static const char start_node_block[] = "{";
    static const char end_node_block1[] = "},";
    static const char end_node_block2[] = "}";
    static const char start_attr_block1[] = "\"attr\": {";
    static const char start_attr_block2[] = "\"param\": {";
    static const char start_attr_block3[] = "\"attrs\": {";
    static const char end_attr_block[] = "},";

    std::ifstream is(model_file.c_str(), std::ios::in);
    if (!is.is_open())
    {
        //LOG_ERROR() << "Cannot open the json file: " << model_file.c_str() << "\n";
        // set_tengine_errno(ENOENT);
        return -1;
    }

    MxnetNode node;
    unsigned int cnt_unknown_name = 1;
    unsigned int line_no = 0;
    int ret = 0;

    // Parse all the nodes in the json file
    while (!is.eof())
    {
        std::string line;
        std::string param, value;
        std::string attr_param, attr_value;

        getline(is, line);
        line_no++;

        // Erase the leading and trailing whitespaces of the line
        Trim(line, " \t\f\v\n\r");

        if (line.empty())
            continue;

        if ((nest == OUT_DEF_BODY && line == start_def_body) || (nest == IN_DEF_BODY && line == start_nodes_list))
        {
            nest++;
            continue;
        }
        else if (nest == IN_NODES_LIST)
        {
            if (line == end_nodes_list)
            {
                nest--;
                if (nest != IN_DEF_BODY)
                    ret = -1;
                break; // Finish parsing all the nodes, end loop
            }
            else if (line == start_node_block)
            {
                node = MxnetNode();
                nest++;
            }
            else
            {
                //LOG_ERROR() << "Parse the json file error in line under mxnert" << line_no << "\n";
                ret = -1;
            }
            continue;
        }
        else if (nest == IN_NODE_BLOCK)
        {
            if (line == end_node_block1 || line == end_node_block2)
            {
                // create a new node
                if (node.name.empty())
                {
                    std::ostringstream unknown;
                    unknown << "unknown" << cnt_unknown_name << std::endl;

                    node.name = unknown.str();
                    cnt_unknown_name++;
                }

                // if(node.op == "Flatten" || node.op == "SliceChannel")
                //    node.op = "Dropout";

                nodelist.push_back(node);
                nest--;
                continue;
            }
            else if (line == start_attr_block1 || line == start_attr_block2 || line == start_attr_block3)
            {
                nest++;
                continue;
            }
            else
            {
                if (!ParseNodeParam(line, param, value) || value.empty())
                {
                    //LOG_ERROR() << "Parse the json file error in line under mxnert " << line_no << "\n";
                    ret = -1;
                    continue;
                }

                if (param == "op")
                {
                    node.op = value;
                }
                else if (param == "name")
                {
                    node.name = value;
                }
                else if (param == "inputs")
                {
                    ParseInputList(value, node.inputs);
                }
                else if (param == "attr" || param == "param" || param == "attrs")
                {
                    if (value == "{}")
                    {
                        continue;
                    }
                    else if (value == "{")
                    {
                        nest++;
                    }
                    else
                    {
                        Trim(value, "{}");
                        if (!value.empty() && ParseNodeParam(value, attr_param, attr_value))
                        {
                            node.attrs[attr_param] = attr_value;
                        }
                        else
                        {
                            //LOG_ERROR() << "Parse the json file error in line under mxnert" << line_no << "\n";
                            ret = -1;
                        }
                    }
                }
                else if (param == "backward_source_id")
                    ;
                else
                {
                    //LOG_ERROR() << "Parse the json file error in line under mxnert" << line_no << "\n";
                    ret = -1;
                }
                continue;
            }
        }
        else if (nest == IN_ATTR_BLOCK)
        {
            if (line == end_attr_block)
            {
                nest--;
            }
            else if (ParseNodeParam(line, attr_param, attr_value))
            {
                node.attrs[attr_param] = attr_value;
            }
            else
            {
                //LOG_ERROR() << "Parse the json file error in line under mxnert" << line_no << "\n";
                ret = -1;
            }
            continue;
        }
        else
        {
            //LOG_ERROR() << "Parse the json file error in line under mxnert" << line_no << "\n";
            ret = -1;
        }
    }

    is.close();
    return 0;
}

int mxnet_serializer::load_binary_file(std::string model_file, std::vector<MxnetParam>& paramlist)
{
    typedef struct
    {
        uint64_t header;
        uint64_t dummy;
        uint64_t block_num;
    } HeadBlock;

    typedef struct
    {
        uint32_t flag;
        uint32_t stype;
        uint32_t dim_size;
        uint32_t dev_type;
        uint32_t dev_id;
        uint32_t type_flag;
    } DataBlock;

    std::ifstream is(model_file.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open())
    {
        //LOG_ERROR() << "Cannot open the param file: " << model_file.c_str() << "\n";
        // set_tengine_errno(ENOENT);
        return -1;
    }

    HeadBlock header;
    is.read((char*)&header, sizeof(uint64_t) * 3);

    // Get all the dims and raw data
    for (unsigned int i = 0; i < header.block_num; i++)
    {
        DataBlock data;
        MxnetParam param = MxnetParam();

        // Read flag
        is.read((char*)&data.flag, sizeof(uint32_t));

        // Read dim_size
        if (data.flag == 0xF993FAC9)
        {
            // Read stype and dim_size
            is.read((char*)&data.stype, sizeof(uint32_t) * 2);
        }
        else if (data.flag == 0xF993FAC8)
        {
            is.read((char*)&data.dim_size, sizeof(uint32_t));
        }
        else
        {
            data.dim_size = data.flag;
        }

        param.dim_size = data.dim_size;
        param.dims.resize(data.dim_size);
        param.data_len = 1;
        for (unsigned int k = 0; k < data.dim_size; k++)
        {
            int64_t d64;
            uint32_t d32;
            if (data.flag == 0xF993FAC9 || data.flag == 0xF993FAC8)
            {
                is.read((char*)&d64, sizeof(int64_t)); // Read dims
                param.dims.at(k) = (int)d64;
            }
            else
            {
                is.read((char*)&d32, sizeof(uint32_t)); // Read dims
                param.dims.at(k) = (int)d32;
            }
            param.data_len *= param.dims.at(k);
        }
        param.data_len *= sizeof(float);

        // Read dev_type, dev_id and type_flag
        is.read((char*)&data.dev_type, sizeof(uint32_t) * 3);

        param.raw_data = (uint8_t*)std::malloc(param.data_len);
        is.read((char*)param.raw_data, param.data_len);

        paramlist.push_back(param);
    }

    // Get all the names
    uint64_t name_count;
    is.read((char*)&name_count, sizeof(uint64_t)); // Read name count
    for (unsigned int i = 0; i < name_count; i++)
    {
        MxnetParam& param = paramlist.at(i);

        uint64_t name_len;
        is.read((char*)&name_len, sizeof(uint64_t)); // Read name length

        param.name.resize(name_len);
        is.read((char*)param.name.data(), name_len); // Read name string

        pos colon_pos = param.name.find(':');
        if (colon_pos != std::string::npos)
            param.name = param.name.substr(colon_pos + 1);
    }

    is.close();
    return 0;
}

void dump_mx_graph(std::vector<MxnetNode>& nodelist, std::vector<MxnetParam>& paramlist)
{
    for (int i = 0; i < nodelist.size(); ++i)
    {
        MxnetNode node = nodelist[i];
        fprintf(stderr, "%d node, op: %s, name: %s\n", i, node.op.c_str(), node.name.c_str());
    }

    fprintf(stderr, "\n");
    for (int i = 0; i < paramlist.size(); i++)
    {
        MxnetParam param = paramlist[i];
        fprintf(stderr, "%d param, name:%s, dim size:%d, dims:", i, param.name.c_str(), param.dim_size);
        for (auto& dim : param.dims)
        {
            fprintf(stderr, "%d ", dim);
        }
        fprintf(stderr, "\n");
    }
}

int mxnet_serializer::load_model(ir_graph_t* graph, std::string model_file, std::string proto_file)
{
    register_op_load();
    std::vector<MxnetNode> nodelist;
    std::vector<MxnetParam> paramlist;

    fprintf(stderr, "load text file...\n");
    if (load_text_file(model_file, nodelist) < 0)
        return -1;
    fprintf(stderr, "load binary file...\n");
    if (load_binary_file(proto_file, paramlist) < 0)
        return -1;
    if (load_constant_tensor(graph, nodelist, paramlist) < 0)
        return -1;
    if (set_graph_input(graph, nodelist, paramlist) < 0)
        return -1;
    if (load_graph_node(graph, nodelist, paramlist) < 0)
        return -1;
    if (set_graph_output(graph) < 0)
        return -1;

    // fprintf(stderr, "dump ...\n");
    // dump_mx_graph(nodelist, paramlist);

    for (auto& param : paramlist)
    {
        std::free(param.raw_data);
    }

    graph->model_format = MODEL_FORMAT_MXNET;
    graph->graph_layout = TENGINE_LAYOUT_NCHW;
    graph->model_layout = TENGINE_LAYOUT_NCHW;
    return 0;
}

graph_t mxnet_serializer::mxnet2tengine(std::string model_file, std::string proto_file)
{
    fprintf(stderr, "----------mxnet2tengine begin----------\n");

    context_t context = create_context(NULL, 1);
    ir_graph_t* ir_graph = create_ir_graph((struct context*)context);
    if (ir_graph == NULL)
    {
        destroy_context(context);
        return NULL;
    }
    ir_graph->attribute->private_context = 1; // new context

    int ret = load_model(ir_graph, model_file, proto_file);
    if (0 != ret)
    {
        destroy_graph(ir_graph);
        return NULL;
    }
    ir_graph->device = find_default_device();

    fprintf(stderr, "----------mxnet2tengine done.----------\n");
    return ir_graph;
}

static int load_conv(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    conv_param* param = (conv_param*)node->op.param_mem;

    const_iterator cit;
    std::vector<int> v1, v2, v3;

    cit = mxnet_node.attrs.find("kernel");
    if (cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v1);
        param->kernel_h = v1.at(0);
        param->kernel_w = v1.at(1);
    }
    cit = mxnet_node.attrs.find("stride");
    if (cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v2);
        param->stride_h = v2.at(0);
        param->stride_w = v2.at(1);
    }
    cit = mxnet_node.attrs.find("pad");
    if (cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v3);
        param->pad_h0 = v3.at(0);
        param->pad_h1 = v3.at(0);
        param->pad_w0 = v3.at(1);
        param->pad_w1 = v3.at(1);
    }
    cit = mxnet_node.attrs.find("num_group");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        int val;
        ist >> val;
        param->group = val;
    }
    cit = mxnet_node.attrs.find("num_filter");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        int val;
        ist >> val;
        param->output_channel = val;
    }

    return 0;
}

static int load_bn(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    batchnorm_param* param = (batchnorm_param*)node->op.param_mem;

    const_iterator cit = mxnet_node.attrs.find("eps");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        float val;
        ist >> val;
        param->eps = val;
    }
    else
    {
        param->eps = 1e-3f;
    }
    cit = mxnet_node.attrs.find("fix_gamma");
    if (cit != mxnet_node.attrs.end())
    {
        if (cit->second == "true" || cit->second == "True")
        {
            ir_tensor_t* gamma_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
            float* gamma_tensor_buffer = (float*)gamma_tensor->data;
            int data_len = gamma_tensor->dims[0];
            for (int i = 0; i < data_len; i++)
            {
                gamma_tensor_buffer[i] = 1.f;
            }
        }
    }

    param->caffe_flavor = 0;

    return 0;
}

static int load_activation(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    const_iterator act_type = mxnet_node.attrs.find("act_type");
    if (act_type != mxnet_node.attrs.end())
    {
        if (act_type->second == "relu")
        {
            relu_param* param = (relu_param*)node->op.param_mem;
            param->negative_slope = 0.f;
        }
        else if (act_type->second == "tanh")
        {
            if (change_node_op(node, OP_TANH) < 0)
                return -1;
        }
        else if (act_type->second == "sigmoid")
        {
            if (change_node_op(node, OP_SIGMOID) < 0)
                return -1;
        }
        else if (act_type->second == "softmax")
        {
            if (change_node_op(node, OP_SOFTMAX) < 0)
                return -1;
            softmax_param* param = (softmax_param*)node->op.param_mem;
            param->axis = 1;
        }
        else
            return -1;
    }
    return 0;
}

static int load_pooling(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    pool_param* param = (pool_param*)node->op.param_mem;

    // set default param
    param->pad_h0 = 0;
    param->pad_h1 = 0;
    param->pad_w0 = 0;
    param->pad_w1 = 0;
    param->stride_h = 1;
    param->stride_w = 1;

    const_iterator cit;
    std::vector<int> v1, v2, v3;

    cit = mxnet_node.attrs.find("kernel");
    if (cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v1);
        param->kernel_h = v1.at(0);
        param->kernel_w = v1.at(1);
    }
    cit = mxnet_node.attrs.find("stride");
    if (cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v2);
        param->stride_h = v2.at(0);
        param->stride_w = v2.at(1);
    }
    cit = mxnet_node.attrs.find("pad");
    if (cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v3);
        param->pad_h0 = v3.at(0);
        param->pad_h1 = v3.at(0);
        param->pad_w0 = v3.at(1);
        param->pad_w1 = v3.at(1);
    }
    cit = mxnet_node.attrs.find("pool_type");
    if (cit != mxnet_node.attrs.end())
    {
        if (cit->second == "max")
        {
            param->pool_method = POOL_MAX;
        }
        else if (cit->second == "avg")
        {
            param->pool_method = POOL_AVG;
            param->caffe_flavor |= COUNT_INCLUDE_PAD_MSK;
        }
    }
    cit = mxnet_node.attrs.find("count_include_pad");
    if (cit != mxnet_node.attrs.end())
    {
        if (cit->second == "False")
        {
            param->caffe_flavor = 0;
        }
    }
    param->global = 0;
    cit = mxnet_node.attrs.find("global_pool");
    if (cit != mxnet_node.attrs.end())
    {
        if (cit->second == "True")
        {
            param->global = 1;
        }
    }

    param->caffe_flavor = 0;

    cit = mxnet_node.attrs.find("pooling_convention");
    if (cit != mxnet_node.attrs.end())
    {
        if (cit->second == "full")
        {
            param->caffe_flavor = 1;
        }
    }

    return 0;
}

static int load_deconv(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    deconv_param* param = (deconv_param*)node->op.param_mem;
    const_iterator cit;
    std::vector<int> v1, v2, v3;

    cit = mxnet_node.attrs.find("kernel");
    if (cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v1);
        param->kernel_h = v1.at(0);
        param->kernel_w = v1.at(1);
    }
    cit = mxnet_node.attrs.find("stride");
    if (cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v2);
        param->stride_h = v2.at(0);
        param->stride_w = v2.at(1);
    }
    cit = mxnet_node.attrs.find("pad");
    if (cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v3);
        param->pad_h0 = v3.at(0);
        param->pad_h1 = v3.at(0);
        param->pad_w0 = v3.at(1);
        param->pad_w1 = v3.at(1);
    }
    cit = mxnet_node.attrs.find("num_group");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        int val;
        ist >> val;
        param->group = val;
    }
    cit = mxnet_node.attrs.find("num_filter");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        int val;
        ist >> val;
        param->num_output = val;
    }

    return 0;
}

static int load_softmax(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    softmax_param* param = (softmax_param*)node->op.param_mem;
    param->axis = 1;

    return 0;
}

static int load_no_param(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    return 0;
}

static int load_concat(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    concat_param* param = (concat_param*)node->op.param_mem;
    const_iterator cit = mxnet_node.attrs.find("dim");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        int axis;
        ist >> axis;
        param->axis = axis;
    }

    return 0;
}

static int load_elt_scalar(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    eltwise_param* param = (eltwise_param*)node->op.param_mem;
    if (mxnet_node.op == "_minus_scalar")
        param->type = ELT_SUB_SCALAR;
    else if (mxnet_node.op == "_mul_scalar")
        param->type = ELT_PROD_SCALAR;
    else if (mxnet_node.op == "_div_scalar")
        param->type = ELT_DIV;
    else if (mxnet_node.op == "_plus_scalar")
        param->type = ELT_SUM;
    param->caffe_flavor = 0;

    //    StaticTensor* tensor = CreateStaticConstTensor(graph, mxnet_node.name + "_scalar");
    std::string scalar_name = mxnet_node.name + "_scalar";
    int scalar_node_id = add_node_above(graph, node->index, OP_CONST, scalar_name.c_str());
    ir_node_t* scalar_node = get_ir_graph_node(graph, scalar_node_id);
    ir_tensor_t* scalar_tensor = get_ir_graph_tensor(graph, scalar_node->output_tensors[0]);

    std::vector<int> dims{1};
    set_ir_tensor_shape(scalar_tensor, dims.data(), dims.size());

    scalar_tensor->data = sys_malloc(sizeof(float));
    float* mem_buf = (float*)scalar_tensor->data;
    const_iterator cit = mxnet_node.attrs.find("scalar");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        float val;
        ist >> val;
        *mem_buf = val;
    }
    else
    {
        *mem_buf = 1; // default value
    }

    set_ir_node_input_tensor(node, node->input_num, scalar_tensor);

    return 0;
}

static int load_elt_add(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    eltwise_param* param = (eltwise_param*)node->op.param_mem;
    param->type = ELT_SUM;
    param->caffe_flavor = 0;

    return 0;
}

static int load_leaky_relu(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    const_iterator cit1 = mxnet_node.attrs.find("act_type");
    const_iterator cit2 = mxnet_node.attrs.find("slope");

    if (cit1 != mxnet_node.attrs.end())
    {
        if (cit1->second == "prelu")
        {
            if (change_node_op(node, OP_PRELU) < 0)
                return -1;
            return 0;
        }
        else if (cit1->second == "elu" && cit2 != mxnet_node.attrs.end())
        {
            if (change_node_op(node, OP_ELU) < 0)
                return -1;
            elu_param* param = (elu_param*)node->op.param_mem;

            std::istringstream ist(cit2->second);
            ist >> param->alpha;

            return 0;
        }
        else if (cit1->second == "leaky" && cit2 != mxnet_node.attrs.end())
        {
            relu_param* param = (relu_param*)node->op.param_mem;

            std::istringstream ist(cit2->second);
            ist >> param->negative_slope;

            return 0;
        }
    }

    return -1;
}

static int load_fc(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    fc_param* param = (fc_param*)node->op.param_mem;
    const_iterator cit = mxnet_node.attrs.find("num_hidden");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        int val;
        ist >> val;
        param->num_output = val;
    }
    return 0;
}

static int load_reshape(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    reshape_param* param = (reshape_param*)node->op.param_mem;
    const_iterator cit;
    std::vector<int> v1;

    cit = mxnet_node.attrs.find("shape");
    if (cit != mxnet_node.attrs.end())
    {
        if (cit->second.find("[") == std::string::npos && cit->second.find(",") == std::string::npos)
        {
            param->dim_size = 1;
            param->re_shape = (int*)sys_malloc(sizeof(int) * param->dim_size);
            param->re_shape[0] = atoi(cit->second.c_str());
        }
        else
        {
            ParseAttr_n(cit->second, v1);
            param->dim_size = v1.size();
            param->re_shape = (int*)sys_malloc(sizeof(int) * param->dim_size);
            for (unsigned int i = 0; i < v1.size(); ++i)
            {
                param->re_shape[i] = v1.at(i);
                fprintf(stderr, "reshape: %d\n", param->re_shape[i]);
            }
        }
    }

    cit = mxnet_node.attrs.find("reverse");

    if (cit != mxnet_node.attrs.end())
    {
        ParseAttr_n(cit->second, v1);
        param->reverse = v1[0];
    }
    param->is_mxnet = 1;

    return 0;
}

static int load_swap_axis(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    swap_axis_param* param = (swap_axis_param*)node->op.param_mem;
    const_iterator cit;
    cit = mxnet_node.attrs.find("dim1");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        ist >> param->dim_0;
    }
    cit = mxnet_node.attrs.find("dim2");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        ist >> param->dim_1;
    }
    return 0;
}

static int load_clip(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    const_iterator cit1, cit2;
    cit1 = mxnet_node.attrs.find("a_max");
    cit2 = mxnet_node.attrs.find("a_min");
    if (cit1 != mxnet_node.attrs.end() && cit1->second == "6" && cit2 != mxnet_node.attrs.end() && cit2->second == "0")
    {
        if (change_node_op(node, OP_RELU6) < 0)
            return -1;
    }
    else // TODO: real clip load
        return -1;
    return 0;
}

static int load_rnn(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    const_iterator cit = mxnet_node.attrs.find("mode");
    const_iterator cit1 = mxnet_node.attrs.find("state_size");
    int s_size = atoi(cit1->second.c_str());

    if (cit->second == "lstm")
    {
        if (change_node_op(node, OP_LSTM) < 0)
            return -1;
        lstm_param* param = (lstm_param*)node->op.param_mem;
        param->mxnet_flag = 1;

        param->hidden_size = s_size;
        param->cell_size = s_size;
    }
    else if (cit->second == "gru")
    {
        if (change_node_op(node, OP_GRU) < 0)
            return -1;
        gru_param* param = (gru_param*)node->op.param_mem;
        param->mxnet_flag = 1;
        param->hidden_size = s_size;
    }
    else
    {
        fprintf(stderr, "TODO...\n");
    }

    return 0;
}

static int load_permute(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    permute_param* param = (permute_param*)node->op.param_mem;
    const_iterator cit;
    std::vector<int> v1;

    cit = mxnet_node.attrs.find("axes");

    ParseAttr_n(cit->second, v1);

    param->order0 = v1[0];
    param->order1 = v1[1];
    param->order2 = v1[2];
    param->order3 = -2;

    return 0;
}

static int load_upsampling(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    const_iterator cit1, cit2;
    cit1 = mxnet_node.attrs.find("scale");
    cit2 = mxnet_node.attrs.find("sample_type");

    if (cit2 != mxnet_node.attrs.end())
    {
        if (cit2->second == "nearest" || cit2->second == "bilinear")
        {
            interp_param* param = (interp_param*)node->op.param_mem;

            std::istringstream ist(cit1->second);
            ist >> param->height_scale;
            param->width_scale = param->height_scale;
            if (cit2->second == "nearest")
                param->resize_type = 1;
            else
                param->resize_type = 2;
        }
    }
    else
    {
        if (change_node_op(node, OP_UPSAMPLE) < 0)
            return -1;
        upsample_param* param = (upsample_param*)node->op.param_mem;
        param->scale = atoi(cit1->second.c_str());
    }

    return 0;
}

static int load_crop(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    crop_param* param = (crop_param*)node->op.param_mem;
    const_iterator cit1, cit2, cit3;
    cit1 = mxnet_node.attrs.find("h_w");
    cit2 = mxnet_node.attrs.find("offset");
    cit3 = mxnet_node.attrs.find("num_args");
    param->num_args = atoi(cit3->second.c_str());
    param->flag = 1;
    std::vector<int> v1;
    std::vector<int> v2;
    if (cit1 != mxnet_node.attrs.end())
    {
        ParseAttr_n(cit1->second, v1);
        if (v1.size() == 1)
        {
            param->crop_h = v1.at(0);
            param->crop_w = v1.at(0);
        }
        if (v1.size() == 2)
        {
            param->crop_h = v1.at(0);
            param->crop_w = v1.at(1);
        }
    }
    if (cit2 != mxnet_node.attrs.end())
    {
        ParseAttr_n(cit2->second, v2);
        param->offset_h = v2.at(0);
        param->offset_w = v2.at(0);
        param->offset_c = 0;
    }

    return 0;
}

static int load_flatten(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    // source code did not set any param?
    return 0;
}

static int load_embedding(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    embedding_param* param = (embedding_param*)node->op.param_mem;
    const_iterator cit1 = mxnet_node.attrs.find("input_dim");
    const_iterator cit2 = mxnet_node.attrs.find("output_dim");
    param->input_dim = atoi(cit1->second.c_str());
    param->num_output = atoi(cit2->second.c_str());
    param->weight_data_size = param->input_dim * param->num_output;

    return 0;
}

static int load_instance_norm(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    instancenorm_Param* param = (instancenorm_Param*)node->op.param_mem;
    const_iterator cit = mxnet_node.attrs.find("eps");
    if (cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        float val;
        ist >> val;
        param->eps = val;
    }
    else
    {
        param->eps = 1e-3f;
    }
    return 0;
}

static int load_psroipooling(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    psroipooling_param* param = (psroipooling_param*)node->op.param_mem;
    const_iterator cit1, cit2, cit3;
    cit1 = mxnet_node.attrs.find("pooled_size");
    cit2 = mxnet_node.attrs.find("spatial_scale");
    cit3 = mxnet_node.attrs.find("output_dim");

    if (cit1 != mxnet_node.attrs.end())
    {
        param->pooled_w = atoi(cit1->second.c_str());
        param->pooled_h = atoi(cit1->second.c_str());
    }
    if (cit2 != mxnet_node.attrs.end())
    {
        param->spatial_scale = atoi(cit2->second.c_str());
    }
    else
    {
        param->spatial_scale = 1;
    }

    if (cit3 != mxnet_node.attrs.end())
    {
        param->spatial_scale = atoi(cit3->second.c_str());
    }
    else
    {
        param->output_dim = 1;
    }

    return 0;
}

static int load_roi_align(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    roialign_param* param = (roialign_param*)node->op.param_mem;
    const_iterator cit1, cit2;
    cit1 = mxnet_node.attrs.find("pooled_size");
    cit2 = mxnet_node.attrs.find("spatial_scale");

    std::vector<int> v1;

    if (cit1 != mxnet_node.attrs.end())
    {
        printf("Into sreialize\n");
        ParseAttr_n(cit1->second, v1);
        if (v1.size() == 1)
        {
            param->pooled_width = v1.at(0);
            param->pooled_height = v1.at(0);
        }
        if (v1.size() == 2)
        {
            param->pooled_width = v1.at(0);
            param->pooled_height = v1.at(1);
        }
    }
    if (cit2 != mxnet_node.attrs.end())
    {
        param->spatial_scale = atoi(cit2->second.c_str());
    }
    else
    {
        param->spatial_scale = 1;
    }

    return 0;
}

static int load_unary(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    unary_param* param = (unary_param*)node->op.param_mem;
    const std::string& op_name = mxnet_node.op;

    if (op_name == "abs")
    {
        param->type = 0;
    }
    else if (op_name == "neg")
    {
        param->type = 1;
    }
    else if (op_name == "floor")
    {
        param->type = 3;
    }
    else if (op_name == "ceil")
    {
        param->type = 3;
    }
    else if (op_name == "log")
    {
        param->type = 8;
    }
    else if (op_name == "cos")
    {
        param->type = 10;
    }
    else if (op_name == "asin")
    {
        param->type = 12;
    }
    else if (op_name == "acos")
    {
        param->type = 13;
    }
    else if (op_name == "atan")
    {
        param->type = 14;
    }
    else if (op_name == "reciprocal")
    {
        param->type = 15;
    }
    return 0;
}

static int load_elt_mul(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    eltwise_param* param = (eltwise_param*)node->op.param_mem;
    param->type = ELT_PROD;
    param->caffe_flavor = 0;

    return 0;
}

static int load_spatial_transformer(ir_graph_t* graph, ir_node_t* node, const MxnetNode& mxnet_node)
{
    spatialtransformer_param* param = (spatialtransformer_param*)node->op.param_mem;
    const_iterator cit1, cit2, cit3;
    cit1 = mxnet_node.attrs.find("sampler_type");
    cit2 = mxnet_node.attrs.find("target_shape");
    cit3 = mxnet_node.attrs.find("transform_type");

    std::vector<int> v1;

    if (cit1 != mxnet_node.attrs.end())
    {
        if (cit1->second == "bilinear")
        {
            param->sampler_type = 1;
        }
        else if (cit1->second == "nearest")
        {
            param->sampler_type = 0;
        }
        else
        {
            param->sampler_type = -1;
        }
    }
    if (cit2 != mxnet_node.attrs.end())
    {
        ParseAttr(cit2->second, v1);
        param->target_shape = (int*)sys_malloc(sizeof(int) * 2);
        param->target_shape[0] = v1.at(0);
        param->target_shape[1] = v1.at(1);
    }
    if (cit3 != mxnet_node.attrs.end())
    {
        if (cit3->second == "affine")
            param->transformer_type = 0;
        else
            param->transformer_type = -1;
    }

    return 0;
}

void mxnet_serializer::register_op_load()
{
    op_load_map["Activation"] = std::pair<int, op_load_t>(OP_RELU, load_activation);
    op_load_map["BatchNorm"] = std::pair<int, op_load_t>(OP_BATCHNORM, load_bn);
    op_load_map["Convolution"] = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["Deconvolution"] = std::pair<int, op_load_t>(OP_DECONV, load_deconv);
    op_load_map["Pooling"] = std::pair<int, op_load_t>(OP_POOL, load_pooling);
    op_load_map["SoftmaxOutput"] = std::pair<int, op_load_t>(OP_SOFTMAX, load_softmax);
    op_load_map["SoftmaxActivation"] = std::pair<int, op_load_t>(OP_SOFTMAX, load_softmax);
    op_load_map["Concat"] = std::pair<int, op_load_t>(OP_CONCAT, load_concat);
    op_load_map["Dropout"] = std::pair<int, op_load_t>(OP_DROPOUT, load_no_param);
    op_load_map["_minus_scalar"] = std::pair<int, op_load_t>(OP_ELTWISE, load_elt_scalar);
    op_load_map["_mul_scalar"] = std::pair<int, op_load_t>(OP_ELTWISE, load_elt_scalar);
    op_load_map["elemwise_add"] = std::pair<int, op_load_t>(OP_ELTWISE, load_elt_add);
    op_load_map["LeakyReLU"] = std::pair<int, op_load_t>(OP_RELU, load_leaky_relu);
    op_load_map["FullyConnected"] = std::pair<int, op_load_t>(OP_FC, load_fc);
    op_load_map["Reshape"] = std::pair<int, op_load_t>(OP_RESHAPE, load_reshape);
    op_load_map["SwapAxis"] = std::pair<int, op_load_t>(OP_SWAP_AXIS, load_swap_axis);
    op_load_map["add_n"] = std::pair<int, op_load_t>(OP_ADD_N, load_no_param);
    op_load_map["clip"] = std::pair<int, op_load_t>(OP_CLIP, load_clip);
    op_load_map["RNN"] = std::pair<int, op_load_t>(OP_RNN, load_rnn);
    op_load_map["transpose"] = std::pair<int, op_load_t>(OP_PERMUTE, load_permute);
    op_load_map["UpSampling"] = std::pair<int, op_load_t>(OP_INTERP, load_upsampling);
    op_load_map["Crop"] = std::pair<int, op_load_t>(OP_CROP, load_crop);
    op_load_map["Copy"] = std::pair<int, op_load_t>(OP_DROPOUT, load_no_param);
    op_load_map["Flatten"] = std::pair<int, op_load_t>(OP_FLATTEN, load_flatten);
    op_load_map["Embedding"] = std::pair<int, op_load_t>(OP_EMBEDDING, load_embedding);
    op_load_map["InstanceNorm"] = std::pair<int, op_load_t>(OP_INSTANCENORM, load_instance_norm);
    //    op_load_map["Reduction"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduction);
    op_load_map["_contrib_PSROIPooling"] = std::pair<int, op_load_t>(OP_PSROIPOOLING, load_psroipooling);
    op_load_map["_contrib_ROIAlign"] = std::pair<int, op_load_t>(OP_ROIALIGN, load_roi_align);
    op_load_map["abs"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["neg"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["ceil"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["floor"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["sin"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["cos"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["atan"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["reciprocal"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["tan"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["broadcast_mul"] = std::pair<int, op_load_t>(OP_BROADMUL, load_no_param);
    op_load_map["_div_scalar"] = std::pair<int, op_load_t>(OP_ELTWISE, load_elt_scalar);
    op_load_map["_plus_scalar"] = std::pair<int, op_load_t>(OP_ELTWISE, load_elt_scalar);
    op_load_map["elemwise_mul"] = std::pair<int, op_load_t>(OP_ELTWISE, load_elt_mul);
    op_load_map["L2Normalization"] = std::pair<int, op_load_t>(OP_L2NORMALIZATION, load_no_param);
    op_load_map["SpatialTransformer"] = std::pair<int, op_load_t>(OP_SPATIALTRANSFORMER, load_spatial_transformer);
}