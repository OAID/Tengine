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
 * Copyright (c) 2017, Open AI Lab
 * Author: jingyou@openailab.com
 */

#include "mxnet_serializer.hpp"

#include "type_name.hpp"
#include "data_type.hpp"
#include "tengine_errno.hpp"
#include "static_graph.hpp"
#include "operator_manager.hpp"
#include "operator/conv_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/softmax_param.hpp"
#include "operator/batch_norm_param.hpp"
#include "operator/relu_param.hpp"
#include "operator/eltwise_param.hpp"
#include "operator/fc_param.hpp"
#include "operator/reshape_param.hpp"

//#define DEBUG

namespace TEngine {

typedef std::string::size_type pos;
typedef std::map<std::string, std::string>::const_iterator const_iterator;
using op_load_t = std::function<bool(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)>;

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
    if(colon_pos != std::string::npos)
    {
        param = str.substr(0, colon_pos);
        Trim(param, " \t\f\v\n\r\",");

        value = str.substr(colon_pos + 1);
        Trim(value, " \t\f\v\n\r\",");

        return true;
    }
    return false;
}

static void ParseInputList(const std::string& str, std::vector<int>& inputs)
{
    // Remove leading '[' and trailing ']'
    std::string s = str.substr(1, str.length() - 2);

    if(s.empty())
        return;

    pos bracket_pos = s.find('[');
    while(bracket_pos != std::string::npos)
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
    for(unsigned int i = 0; i < inputs.size(); i++)
        std::cout << inputs.at(i) << " ";
    std::cout << std::endl;
#endif
}

bool MxnetSerializer::LoadTextFile(const char* fname, std::vector<MxnetNode>& nodelist)
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

    std::ifstream is(fname, std::ios::in);
    if(!is.is_open())
    {
        LOG_ERROR() << "Cannot open the json file: " << fname << "\n";
        set_tengine_errno(ENOENT);
        return false;
    }

    MxnetNode node;
    unsigned int cnt_unknown_name = 1;
    unsigned int line_no = 0;
    bool ret = true;

    // Parse all the nodes in the json file
    while(!is.eof())
    {
        std::string line;
        std::string param, value;
        std::string attr_param, attr_value;

        getline(is, line);
        line_no++;

        // Erase the leading and trailing whitespaces of the line
        Trim(line, " \t\f\v\n\r");

        if(line.empty())
            continue;

        if((nest == OUT_DEF_BODY && line == start_def_body) || (nest == IN_DEF_BODY && line == start_nodes_list))
        {
            nest++;
            continue;
        }
        else if(nest == IN_NODES_LIST)
        {
            if(line == end_nodes_list)
            {
                nest--;
                if(nest != IN_DEF_BODY)
                    ret = false;
                break;    // Finish parsing all the nodes, end loop
            }
            else if(line == start_node_block)
            {
                node = MxnetNode();
                nest++;
            }
            else
            {
                LOG_ERROR() << "Parse the json file error in line " << line_no << "\n";
                ret = false;
            }
            continue;
        }
        else if(nest == IN_NODE_BLOCK)
        {
            if(line == end_node_block1 || line == end_node_block2)
            {
                // create a new node
                if(node.name.empty())
                {
                    std::ostringstream unknown;
                    unknown << "unknown" << cnt_unknown_name << std::endl;

                    node.name = unknown.str();
                    cnt_unknown_name++;
                }
                if(node.op == "Flatten")
                    node.op = "Dropout";
                nodelist.push_back(node);
                nest--;
                continue;
            }
            else if(line == start_attr_block1 || line == start_attr_block2 || line == start_attr_block3)
            {
                nest++;
                continue;
            }
            else
            {
                if(!ParseNodeParam(line, param, value) || value.empty())
                {
                    LOG_ERROR() << "Parse the json file error in line " << line_no << "\n";
                    ret = false;
                    continue;
                }

                if(param == "op")
                {
                    node.op = value;
                }
                else if(param == "name")
                {
                    node.name = value;
                }
                else if(param == "inputs")
                {
                    ParseInputList(value, node.inputs);
                }
                else if(param == "attr" || param == "param" || param == "attrs")
                {
                    if(value == "{}")
                    {
                        continue;
                    }
                    else if(value == "{")
                    {
                        nest++;
                    }
                    else
                    {
                        Trim(value, "{}");
                        if(!value.empty() && ParseNodeParam(value, attr_param, attr_value))
                        {
                            node.attrs[attr_param] = attr_value;
                        }
                        else
                        {
                            LOG_ERROR() << "Parse the json file error in line " << line_no << "\n";
                            ret = false;
                        }
                    }
                }
                else if(param == "backward_source_id")
                    ;
                else
                {
                    LOG_ERROR() << "Parse the json file error in line " << line_no << "\n";
                    ret = false;
                }
                continue;
            }
        }
        else if(nest == IN_ATTR_BLOCK)
        {
            if(line == end_attr_block)
            {
                nest--;
            }
            else if(ParseNodeParam(line, attr_param, attr_value))
            {
                node.attrs[attr_param] = attr_value;
            }
            else
            {
                LOG_ERROR() << "Parse the json file error in line " << line_no << "\n";
                ret = false;
            }
            continue;
        }
        else
        {
            LOG_ERROR() << "Parse the json file error in line " << line_no << "\n";
            ret = false;
        }
    }

    is.close();
    return ret;
}

bool MxnetSerializer::LoadBinaryFile(const char* fname, std::vector<MxnetParam>& paramlist)
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

    std::ifstream is(fname, std::ios::in | std::ios::binary);
    if(!is.is_open())
    {
        LOG_ERROR() << "Cannot open the param file: " << fname << "\n";
        set_tengine_errno(ENOENT);
        return false;
    }

    HeadBlock header;
    is.read(( char* )&header, sizeof(uint64_t) * 3);

    // Get all the dims and raw data
    for(unsigned int i = 0; i < header.block_num; i++)
    {
        DataBlock data;
        MxnetParam param = MxnetParam();

        // Read flag
        is.read(( char* )&data.flag, sizeof(uint32_t));

        // Read dim_size
        if(data.flag == 0xF993FAC9)
        {
            // Read stype and dim_size
            is.read(( char* )&data.stype, sizeof(uint32_t) * 2);
        }
        else if(data.flag == 0xF993FAC8)
        {
            is.read(( char* )&data.dim_size, sizeof(uint32_t));
        }
        else
        {
            data.dim_size = data.flag;
        }

        param.dim_size = data.dim_size;
        param.dims.resize(data.dim_size);
        param.data_len = 1;
        for(unsigned int k = 0; k < data.dim_size; k++)
        {
            int64_t d64;
            uint32_t d32;
            if(data.flag == 0xF993FAC9 || data.flag == 0xF993FAC8)
            {
                is.read(( char* )&d64, sizeof(int64_t));    // Read dims
                param.dims.at(k) = ( int )d64;
            }
            else
            {
                is.read(( char* )&d32, sizeof(uint32_t));    // Read dims
                param.dims.at(k) = ( int )d32;
            }
            param.data_len *= param.dims.at(k);
        }
        param.data_len *= sizeof(float);

        // Read dev_type, dev_id and type_flag
        is.read(( char* )&data.dev_type, sizeof(uint32_t) * 3);

        param.raw_data = ( uint8_t* )std::malloc(param.data_len);
        is.read(( char* )param.raw_data, param.data_len);

        paramlist.push_back(param);
    }

    // Get all the names
    uint64_t name_count;
    is.read(( char* )&name_count, sizeof(uint64_t));    // Read name count
    for(unsigned int i = 0; i < name_count; i++)
    {
        MxnetParam& param = paramlist.at(i);

        uint64_t name_len;
        is.read(( char* )&name_len, sizeof(uint64_t));    // Read name length

        param.name.resize(name_len);
        is.read(( char* )param.name.data(), name_len);    // Read name string

        pos colon_pos = param.name.find(':');
        if(colon_pos != std::string::npos)
            param.name = param.name.substr(colon_pos + 1);
    }

#ifdef DEBUG
    std::cout << "Dump Param List: " << paramlist.size() << std::endl;
    for(unsigned int i = 0; i < paramlist.size(); i++)
    {
        std::cout << "    Name: " << paramlist.at(i).name << std::endl;
        std::cout << "    dim_size: " << paramlist.at(i).dim_size << std::endl;
        std::cout << "    data_len: " << paramlist.at(i).data_len << std::endl;
        std::cout << "    raw_data: \n";
        for(int j = 0; j < paramlist.at(i).data_len; j++)
            printf("%#x, ", paramlist.at(i).raw_data[j]);
        printf("\n");
    }
#endif

    is.close();
    return true;
}

bool MxnetSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    if(file_list.size() != GetFileNum())
        return false;

    std::vector<MxnetNode> nodelist;
    if(!LoadTextFile(file_list[0].c_str(), nodelist))
    {
        LOG_ERROR() << "Parse text file " << file_list[0].c_str() << " failed\n";
        return false;
    }

    std::vector<MxnetParam> paramlist;
    if(!LoadBinaryFile(file_list[1].c_str(), paramlist))
    {
        LOG_ERROR() << "Parse binary file " << file_list[1].c_str() << " failed\n";
        return false;
    }

    SetGraphSource(graph, file_list[1]);
    SetGraphSourceFormat(graph, "mxnet");
    SetGraphConstTensorFile(graph, file_list[1]);

    return LoadGraph(graph, nodelist, paramlist);
}

bool MxnetSerializer::LoadConstTensor(StaticGraph* graph, const std::vector<MxnetParam>& paramlist)
{
    int const_tensor_number = paramlist.size();

#ifdef DEBUG
    std::cout << "Load Const Tensor:" << std::endl;
#endif

    for(int i = 0; i < const_tensor_number; i++)
    {
        const MxnetParam& mxnet_tensor = paramlist.at(i);

#ifdef DEBUG
        std::cout << "    name: " << mxnet_tensor.name << std::endl;
#endif

        StaticTensor* tensor = CreateStaticConstTensor(graph, mxnet_tensor.name);

        std::vector<int> dims = mxnet_tensor.dims;
        SetTensorDim(tensor, dims);

        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
        SetTensorSize(tensor, mxnet_tensor.data_len);

        uint8_t* mem_buf = ( uint8_t* )std::malloc(mxnet_tensor.data_len);
        uint8_t* raw_data = ( uint8_t* )mxnet_tensor.raw_data;

        /* load data */
        for(int k = 0; k < mxnet_tensor.data_len; k++)
            mem_buf[k] = raw_data[k];

        SetConstTensorBuffer(tensor, mem_buf);
        SetConstTensorFileLocation(tensor, -1, 0);

        /* Now, create the node .... */
        StaticOp* op = CreateStaticOp(graph, "Const");
        StaticNode* node = CreateStaticNode(graph, GetTensorName(tensor));

        SetNodeOp(node, op);

        AddNodeOutputTensor(node, tensor);
    }

    return true;
}

static bool GetParam(const std::string name, const std::vector<MxnetParam>& paramlist, MxnetParam& param)
{
    for(unsigned int i = 0; i < paramlist.size(); i++)
    {
        if(paramlist.at(i).name == name)
        {
            param = paramlist.at(i);
            return true;
        }
    }
    return false;
}

void MxnetSerializer::CreateInputNode(StaticGraph* graph, const std::vector<MxnetNode>& nodelist,
                                      const std::vector<MxnetParam>& paramlist)
{
#ifdef DEBUG
    std::cout << "Create Input Node:" << std::endl;
#endif

    for(unsigned int i = 0; i < nodelist.size(); i++)
    {
        const MxnetNode& mxnet_node = nodelist.at(i);
        if(mxnet_node.name == "data")
        {
            if(FindConstTensor(graph, mxnet_node.name) != nullptr)
                continue;

            // Create an input tensor
            StaticTensor* tensor = CreateStaticTensor(graph, mxnet_node.name);

            SetTensorDataType(tensor, DataType::GetTypeID("float32"));

            SetTensorDataLayout(tensor, "NCHW");

            MxnetParam param;
            if(GetParam(mxnet_node.name, paramlist, param))
            {
                std::vector<int> dims = param.dims;
                SetTensorDim(tensor, dims);
            }

            StaticNode* node = CreateStaticNode(graph, mxnet_node.name);
            StaticOp* op = CreateStaticOp(graph, "InputOp");

            SetNodeOp(node, op);

            AddNodeOutputTensor(node, tensor);

            /*add this node into graph input node list */
            AddGraphInputNode(graph, node);
#ifdef DEBUG
            std::cout << "    create an input node for " << mxnet_node.name << std::endl;
#endif
        }
    }
}

void MxnetSerializer::LoadNode(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node,
                               const std::vector<MxnetNode>& nodelist)
{
    int input_number = mxnet_node.inputs.size();

#ifdef DEBUG
    std::cout << "name: " << mxnet_node.name << " input_number: " << input_number << std::endl;
#endif

    for(int i = 0; i < input_number; i++)
    {
        int input_idx = mxnet_node.inputs.at(i);
        const MxnetNode& input_node = nodelist.at(input_idx);
        if(input_node.name == "prob_label")
            continue;

#ifdef DEBUG
        std::cout << "    input: " << input_idx << " " << input_node.name << std::endl;
#endif

        StaticTensor* tensor = FindTensor(graph, input_node.name);
        if(input_node.op == "null")
        {
            if(mxnet_node.op == "BatchNorm" || mxnet_node.op == "LeakyReLU")
            {
                SetTensorDataLayout(tensor, "W");
            }
            else if(mxnet_node.op == "FullyConnected")
            {
                if(i == 1)    // weight
                    SetTensorDataLayout(tensor, "HW");
                else if(i == 2)    // bias
                    SetTensorDataLayout(tensor, "W");
            }
            else
            {
                if(i == 1)    // weight
                    SetTensorDataLayout(tensor, "NCHW");
                else if(i == 2)    // bias
                    SetTensorDataLayout(tensor, "W");
            }
        }

        AddNodeInputTensor(node, tensor);
    }

    const std::string& output_name = mxnet_node.name;

    StaticTensor* tensor = CreateStaticTensor(graph, output_name);

    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    SetTensorDataLayout(tensor, "NCHW");
    AddNodeOutputTensor(node, tensor);
}

bool MxnetSerializer::LoadGraph(StaticGraph* graph, const std::vector<MxnetNode>& nodelist,
                                const std::vector<MxnetParam>& paramlist)
{
    SetGraphIdentity(graph, "mxnet", graph->model_name, "0");

    LoadConstTensor(graph, paramlist);
    CreateInputNode(graph, nodelist, paramlist);

    unsigned int i;
    for(i = 0; i < nodelist.size(); i++)
    {
        MxnetNode mxnet_node = nodelist.at(i);

        if(mxnet_node.op == "null")
            continue;

        if(!FindOpLoadMethod(mxnet_node.op))
        {
            LOG_ERROR() << "Cannot find load function for operator: " << mxnet_node.op << "\n";
            break;
        }

        StaticNode* node = CreateStaticNode(graph, mxnet_node.name);

        LoadNode(graph, node, mxnet_node, nodelist);

        op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(mxnet_node.op));

        if(!op_func(graph, node, mxnet_node))
            break;
    }

    if(i < nodelist.size())
        return false;

#ifdef DEBUG
    std::cout << "Successfully load all nodes" << std::endl;
#endif

    return true;
}

static bool LoadMxnetSoftmax(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    StaticOp* op = CreateStaticOp(graph, "Softmax");

    SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam("Softmax"));

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    AddGraphOutputNode(graph, node);

    return true;
}

static bool LoadMxnetConcat(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    StaticOp* op = CreateStaticOp(graph, "Concat");

    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));

    SetOperatorParam(op, param);

    SetNodeOp(node, op);

    return true;
}

// Parse "(xxx, yyy)"
static void ParseAttr(const std::string str, std::vector<int>& result)
{
    // Remove leading '(' and trailing ')'
    std::string s = str.substr(1, str.length() - 2);

    pos comma_pos = s.find(',');
    std::string s1 = s.substr(0, comma_pos);
    std::string s2 = s.substr(comma_pos + 1);
    s2.erase(0, s2.find_first_not_of(" "));

    std::istringstream ist1(s1);
    std::istringstream ist2(s2);
    int i, j;
    ist1 >> i;
    ist2 >> j;
    result.push_back(i);
    result.push_back(j);
}

static bool LoadMxnetConvolution(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));

    const_iterator cit;
    std::vector<int> v1, v2, v3;

    cit = mxnet_node.attrs.find("kernel");
    if(cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v1);
        param.kernel_h = v1.at(0);
        param.kernel_w = v1.at(1);
    }
    cit = mxnet_node.attrs.find("stride");
    if(cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v2);
        param.stride_h = v2.at(0);
        param.stride_w = v2.at(1);
    }
    cit = mxnet_node.attrs.find("pad");
    if(cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v3);
        param.pad_h = v3.at(0);
        param.pad_w = v3.at(1);
    }
    cit = mxnet_node.attrs.find("num_group");
    if(cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        int val;
        ist >> val;
        param.group = val;
    }
    cit = mxnet_node.attrs.find("num_filter");
    if(cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        int val;
        ist >> val;
        param.output_channel = val;
    }

#ifdef DEBUG
    std::cout << "ConvParam : " << param.kernel_h << ", " << param.kernel_w << ", " << param.stride_h << ", "
              << param.stride_w << ", " << param.pad_h << ", " << param.pad_w << ", " << param.group << ", "
              << param.output_channel << std::endl;
#endif

    StaticOp* op = CreateStaticOp(graph, "Convolution");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetPooling(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));

    const_iterator cit;
    std::vector<int> v1, v2, v3;

    cit = mxnet_node.attrs.find("kernel");
    if(cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v1);
        param.kernel_h = v1.at(0);
        param.kernel_w = v1.at(1);

        param.kernel_shape.resize(2);
        param.kernel_shape[0] = param.kernel_h;
        param.kernel_shape[1] = param.kernel_w;
    }
    cit = mxnet_node.attrs.find("stride");
    if(cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v2);
        param.stride_h = v2.at(0);
        param.stride_w = v2.at(1);

        param.strides.resize(2);
        param.strides[0] = param.stride_h;
        param.strides[1] = param.stride_w;
    }
    cit = mxnet_node.attrs.find("pad");
    if(cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v3);
        param.pad_h = v3.at(0);
        param.pad_w = v3.at(1);

        param.pads.resize(4);
        param.pads[0] = param.pad_h;
        param.pads[1] = param.pad_w;
        param.pads[2] = param.pad_h;
        param.pads[3] = param.pad_w;
    }
    cit = mxnet_node.attrs.find("pool_type");
    if(cit != mxnet_node.attrs.end())
    {
        if(cit->second == "max")
        {
            param.global = 0;
            param.alg = kPoolMax;
        }
        else if(cit->second == "avg")
        {
            param.global = 1;
            param.alg = kPoolAvg;
        }
    }
    param.caffe_flavor = 0;

#ifdef DEBUG
    std::cout << "PoolParam : " << param.kernel_h << ", " << param.kernel_w << ", " << param.stride_h << ", "
              << param.stride_w << ", " << param.pad_h << ", " << param.pad_w << ", " << param.global << ", "
              << param.alg << std::endl;
#endif

    StaticOp* op = CreateStaticOp(graph, "Pooling");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetBatchNorm(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    BatchNormParam param = any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));

    const_iterator cit = mxnet_node.attrs.find("eps");
    if(cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        float val;
        ist >> val;
        param.eps = val;
    }
    else
    {
        param.eps=1e-3f;
    }

    param.caffe_flavor = 0;

#ifdef DEBUG
    std::cout << "BatchNormParam : " << param.eps << std::endl;
#endif

    StaticOp* op = CreateStaticOp(graph, "BatchNormalization");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetDropout(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    StaticOp* op = CreateStaticOp(graph, "Dropout");

    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetRelu(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));
    param.negative_slope = 0.f;

    StaticOp* op = CreateStaticOp(graph, "ReLu");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetEltScalar(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
    if(mxnet_node.op == "_minus_scalar")
        param.type = ELT_SUB_SCALAR;
    else if(mxnet_node.op == "_mul_scalar")
        param.type = ELT_PROD_SCALAR;
    param.caffe_flavor = 0;

    StaticTensor* tensor = CreateStaticConstTensor(graph, mxnet_node.name + "_scalar");
    std::vector<int> dims;
    dims.push_back(1);
    SetTensorDim(tensor, dims);
    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    SetTensorDataLayout(tensor, "W");
    SetTensorSize(tensor, sizeof(float));

    float* mem_buf = ( float* )std::malloc(sizeof(float));
    const_iterator cit = mxnet_node.attrs.find("scalar");
    if(cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        float val;
        ist >> val;
        *mem_buf = val;
    }
    SetConstTensorBuffer(tensor, mem_buf);
    SetConstTensorFileLocation(tensor, -1, 0);

    StaticOp* op = CreateStaticOp(graph, "Const");
    StaticNode* new_node = CreateStaticNode(graph, GetTensorName(tensor));
    SetNodeOp(new_node, op);
    AddNodeOutputTensor(new_node, tensor);

    AddNodeInputTensor(node, tensor);

    op = CreateStaticOp(graph, "Eltwise");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetElemwiseAdd(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
    param.type = ELT_SUM;
    param.caffe_flavor = 0;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetLeakyReLU(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    const_iterator cit = mxnet_node.attrs.find("act_type");
    if(cit == mxnet_node.attrs.end() || cit->second != "prelu")
    {
        return false;
    }

    StaticOp* op = CreateStaticOp(graph, "PReLU");
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetFullyConnected(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    FCParam param = any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));
    const_iterator cit = mxnet_node.attrs.find("num_hidden");
    if(cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        int val;
        ist >> val;
        param.num_output = val;
    }

    StaticOp* op = CreateStaticOp(graph, "FullyConnected");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetReshape(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    ReshapeParam param = any_cast<ReshapeParam>(OpManager::GetOpDefParam("Reshape"));

    const_iterator cit;
    std::vector<int> v1;

    cit = mxnet_node.attrs.find("shape");
    if(cit != mxnet_node.attrs.end())
    {
        ParseAttr(cit->second, v1);

        param.dim_3 = -2;
        param.dim_2 = -2;
        param.dim_1 = -2;
        param.dim_0 = -2;

        param.dim_size = v1.size();

        switch(v1.size())
        {
            case 4:
                param.dim_3 = v1.at(3);
            case 3:
                param.dim_2 = v1.at(2);
            case 2:
                param.dim_1 = v1.at(1);
            case 1:
                param.dim_0 = v1.at(0);
                break;
            default:
                return false;
        }
    }

    StaticOp* op = CreateStaticOp(graph, "Reshape");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

bool MxnetSerializerRegisterOpLoader(void)
{
    SerializerPtr serializer;

    if(!SerializerManager::SafeGet("mxnet", serializer))
        return false;

    MxnetSerializer* p_mxnet = dynamic_cast<MxnetSerializer*>(serializer.get());

    p_mxnet->RegisterOpLoadMethod("Convolution", op_load_t(LoadMxnetConvolution));
    p_mxnet->RegisterOpLoadMethod("Pooling", op_load_t(LoadMxnetPooling));
    p_mxnet->RegisterOpLoadMethod("SoftmaxOutput", op_load_t(LoadMxnetSoftmax));
    p_mxnet->RegisterOpLoadMethod("Concat", op_load_t(LoadMxnetConcat));
    p_mxnet->RegisterOpLoadMethod("BatchNorm", op_load_t(LoadMxnetBatchNorm));
    p_mxnet->RegisterOpLoadMethod("Dropout", op_load_t(LoadMxnetDropout));
    p_mxnet->RegisterOpLoadMethod("Activation", op_load_t(LoadMxnetRelu));

    p_mxnet->RegisterOpLoadMethod("_minus_scalar", op_load_t(LoadMxnetEltScalar));
    p_mxnet->RegisterOpLoadMethod("_mul_scalar", op_load_t(LoadMxnetEltScalar));
    p_mxnet->RegisterOpLoadMethod("elemwise_add", op_load_t(LoadMxnetElemwiseAdd));
    p_mxnet->RegisterOpLoadMethod("LeakyReLU", op_load_t(LoadMxnetLeakyReLU));
    p_mxnet->RegisterOpLoadMethod("FullyConnected", op_load_t(LoadMxnetFullyConnected));

    p_mxnet->RegisterOpLoadMethod("Reshape", op_load_t(LoadMxnetReshape));

    return true;
}

}    // namespace TEngine
