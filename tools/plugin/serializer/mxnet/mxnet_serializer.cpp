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

#include <set>
#include <algorithm>

#include "mxnet_serializer.hpp"

#include "tengine_c_api.h"
#include "exec_attr.hpp"
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
#include "operator/swap_axis_param.hpp"
#include "operator/addn_param.hpp"
#include "operator/lstm_param.hpp"
#include "operator/gru_param.hpp"
#include "operator/permute_param.hpp"
#include "operator/crop_param.hpp"
#include "operator/upsample_param.hpp"
#include "operator/deconv_param.hpp"
#include "operator/flatten_param.hpp"
#include "operator/elu_param.hpp"
#include "operator/interp_param.hpp"
#include "operator/reduction_param.hpp"
#include "operator/instancenorm_param.hpp"
#include "operator/embed_param.hpp"
#include "operator/roialign_param.hpp"
#include "operator/psroipooling_param.hpp"
#include "operator/unary_param.hpp"
#include "operator/broadmul.hpp"
//#define DEBUG

namespace TEngine {

typedef std::string::size_type pos;
typedef std::map<std::string, std::string>::const_iterator const_iterator;
using op_load_t = std::function<bool(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)>;

std::vector<int>& split(const std::string& str, char delim, std::vector<int>& elems, bool skip_empty = true)
{
    std::istringstream iss(str);
    for(std::string item; getline(iss, item, delim);)
        if(skip_empty && item.empty())
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

                // if(node.op == "Flatten" || node.op == "SliceChannel")
                //    node.op = "Dropout";

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
    SetGraphLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelFormat(graph, MODEL_FORMAT_MXNET);

    bool res = LoadGraph(graph, nodelist, paramlist);
    for(std::size_t ii = 0; ii < paramlist.size(); ++ii)
    {
        std::free(paramlist[ii].raw_data);
    }

    return res;
}

bool MxnetSerializer::LoadConstTensor(StaticGraph* graph, const std::vector<MxnetNode>& nodelist,
                                      const std::vector<MxnetParam>& paramlist)
{
    int const_tensor_number = paramlist.size();
    std::set<std::string> node_name_set;

    for(unsigned int i = 0; i < nodelist.size(); i++)
    {
        const MxnetNode& mxnet_node = nodelist.at(i);
        node_name_set.insert(mxnet_node.name);
    }

#ifdef DEBUG
    std::cout << "Load Const Tensor:" << std::endl;
#endif

    for(int i = 0; i < const_tensor_number; i++)
    {
        const MxnetParam& mxnet_tensor = paramlist.at(i);

#ifdef DEBUG
        std::cout << "    name: " << mxnet_tensor.name << std::endl;
#endif
        if(!node_name_set.count(mxnet_tensor.name))
        {
            std::cout << "skip tensor: " << mxnet_tensor.name << "\n";
            continue;
        }

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
        if(mxnet_node.name == "data" || mxnet_node.name == "x" || mxnet_node.name == "y")
        {
            if(FindConstTensor(graph, mxnet_node.name) != nullptr)
                continue;

            // Create an input tensor
            StaticTensor* tensor = CreateStaticTensor(graph, mxnet_node.name);

            SetTensorDataType(tensor, DataType::GetTypeID("float32"));

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

        if(input_node.name.find("label") != std::string::npos || input_node.name.find("state") != std::string::npos)
            continue;

#ifdef DEBUG
        std::cout << "    input: " << input_idx << " " << input_node.name << std::endl;
#endif

        StaticTensor* tensor = FindTensor(graph, input_node.name);

        AddNodeInputTensor(node, tensor);
    }

    const std::string& output_name = mxnet_node.name;

    StaticTensor* tensor = CreateStaticTensor(graph, output_name);

    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    AddNodeOutputTensor(node, tensor);
}

bool MxnetSerializer::LoadGraph(StaticGraph* graph, const std::vector<MxnetNode>& nodelist,
                                const std::vector<MxnetParam>& paramlist)
{
    SetGraphIdentity(graph, "mxnet", graph->model_name, "0");

    LoadConstTensor(graph, nodelist, paramlist);
    CreateInputNode(graph, nodelist, paramlist);

    unsigned int i;
    std::vector<std::string> no_supported_op;
    for(i = 0; i < nodelist.size(); i++)
    {
        MxnetNode mxnet_node = nodelist.at(i);
        if(mxnet_node.op == "null" || mxnet_node.op == "_zeros")
            continue;
        if(!FindOpLoadMethod(mxnet_node.op))
        {
            auto it = find(no_supported_op.begin(),no_supported_op.end(),mxnet_node.op);
            if(it != no_supported_op.end())
                no_supported_op.push_back(mxnet_node.op);
        }
    }
    if(no_supported_op.size())
    {
        LOG_ERROR() << "These" <<no_supported_op.size() <<"ops are not supported \n";
        LOG_ERROR() << "{";
        for(int j = 0; j < (int)no_supported_op.size(); j++)
        {
            LOG_ERROR() << no_supported_op[j] <<",";
        }
        LOG_ERROR() << "}\n";
        return false;
    }   
   
    for(i = 0; i < nodelist.size(); i++)
    {
        MxnetNode mxnet_node = nodelist.at(i);

        if(mxnet_node.op == "null" || mxnet_node.op == "_zeros")
            continue;

       // if(!FindOpLoadMethod(mxnet_node.op))
       // {
       //     LOG_ERROR() << "Cannot find load function for operator: " << mxnet_node.op << "\n";
       //     break;
       // }

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
    param.axis = 1;

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

    std::string s1, s2;
    int i;
    while(1)
    {
        pos comma_pos = s.find(',');
        if(comma_pos != std::string::npos)
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
        param.pad_h0 = v3.at(0);
        param.pad_h1 = v3.at(0);
        param.pad_w0 = v3.at(1);
        param.pad_w1 = v3.at(1);
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
              << param.stride_w << ", " << param.pad_h0 << ", " << param.pad_w0 << ", " << param.group << ", "
              << param.output_channel << std::endl;
#endif

    StaticOp* op = CreateStaticOp(graph, "Convolution");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetDeConvolution(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    DeconvParam param = any_cast<DeconvParam>(OpManager::GetOpDefParam("Deconvolution"));

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
        param.pad_h0 = v3.at(0);
        param.pad_h1 = v3.at(0);
        param.pad_w0 = v3.at(1);
        param.pad_w1 = v3.at(1);
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
        param.num_output = val;
    }

#ifdef DEBUG
    std::cout << "DeconvParam : " << param.kernel_h << ", " << param.kernel_w << ", " << param.stride_h << ", "
              << param.stride_w << ", " << param.pad_h0 << ", " << param.pad_w0 << ", " << param.group << ", "
              << param.num_output << std::endl;
#endif

    StaticOp* op = CreateStaticOp(graph, "Deconvolution");
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
        param.pad_h0 = v3.at(0);
        param.pad_h1 = v3.at(0);
        param.pad_w0 = v3.at(1);
        param.pad_w1 = v3.at(1);
    }
    cit = mxnet_node.attrs.find("pool_type");
    if(cit != mxnet_node.attrs.end())
    {
        if(cit->second == "max")
        {
            param.alg = kPoolMax;
        }
        else if(cit->second == "avg")
        {
            param.alg = kPoolAvg;
            param.caffe_flavor |= COUNT_INCLUDE_PAD_MSK;
        }
    }
    cit = mxnet_node.attrs.find("count_include_pad");
    if(cit != mxnet_node.attrs.end())
    {
        if(cit->second == "False")
        {
            param.caffe_flavor = 0;
        }
    }
    param.global = 0;
    cit = mxnet_node.attrs.find("global_pool");
    if(cit != mxnet_node.attrs.end())
    {
        if(cit->second == "True")
        {
            param.global = 1;
        }
    }
    
    param.caffe_flavor = 0;
    
    cit = mxnet_node.attrs.find("pooling_convention");
    if(cit != mxnet_node.attrs.end())
    {
        if(cit->second == "full")
        {
            param.caffe_flavor = 1;
        }
    }

#ifdef DEBUG
    std::cout << "PoolParam : " << param.kernel_h << ", " << param.kernel_w << ", " << param.stride_h << ", "
              << param.stride_w << ", " << param.pad_h0 << ", " << param.pad_w0 << ", " << param.global << ", "
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
        param.eps = 1e-3f;
    }
    cit = mxnet_node.attrs.find("fix_gamma");
    if(cit != mxnet_node.attrs.end())
    {
        if(cit->second == "true" || cit->second == "True")
        {
            StaticTensor* gamma_tensor = GetNodeInputTensor(graph, node, 1);
            float* gamma_tensor_buffer = ( float* )GetConstTensorBuffer(gamma_tensor);
            std::vector<int> dims = GetTensorDim(gamma_tensor);
            int data_len = dims[0];
            for(int i = 0; i < data_len; i++)
            {
                gamma_tensor_buffer[i] = 1.f;
            }
        }
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

static bool LoadMxnetActivation(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    const_iterator act_type = mxnet_node.attrs.find("act_type");
    if(act_type != mxnet_node.attrs.end())
    {
        if(act_type->second == "relu")
        {
            ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));
            param.negative_slope = 0.f;

            StaticOp* op = CreateStaticOp(graph, "ReLu");
            SetOperatorParam(op, param);
            SetNodeOp(node, op);
        }
        else if(act_type->second == "tanh")
        {
            StaticOp* op = CreateStaticOp(graph, "Tanh");
            SetNodeOp(node, op);
        }
        else if(act_type->second == "sigmoid")
        {
            StaticOp* op = CreateStaticOp(graph, "Sigmoid");
            SetNodeOp(node, op);
        }
        else if(act_type->second == "softmax")
        {
            SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam("Softmax"));
            param.axis = 1;

            StaticOp* op = CreateStaticOp(graph, "Softmax");
            SetOperatorParam(op, param);
            SetNodeOp(node, op);
        }
        else
            return false;
    }

    return true;
}

static bool LoadMxnetEltScalar(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
    if(mxnet_node.op == "_minus_scalar")
        param.type = ELT_SUB_SCALAR;
    else if(mxnet_node.op == "_mul_scalar")
        param.type = ELT_PROD_SCALAR;
    else if(mxnet_node.op == "_div_scalar")
        param.type = ELT_DIV;   
    else if(mxnet_node.op == "_plus_scalar")
        param.type = ELT_SUM;           
    param.caffe_flavor = 0;

    StaticTensor* tensor = CreateStaticConstTensor(graph, mxnet_node.name + "_scalar");
    std::vector<int> dims;
    dims.push_back(1);
    SetTensorDim(tensor, dims);
    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
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

static bool LoadMxnetElemwiseMul(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
    param.type = ELT_PROD;
    param.caffe_flavor = 0;

    StaticOp* op = CreateStaticOp(graph, "Eltwise");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetLeakyReLU(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    const_iterator cit1 = mxnet_node.attrs.find("act_type");
    const_iterator cit2 = mxnet_node.attrs.find("slope");

    if(cit1 != mxnet_node.attrs.end())
    {
        if(cit1->second == "prelu")
        {
            StaticOp* op = CreateStaticOp(graph, "PReLU");
            SetNodeOp(node, op);
            return true;
        }
        else if(cit1->second == "elu" && cit2 != mxnet_node.attrs.end())
        {
            EluParam param = any_cast<EluParam>(OpManager::GetOpDefParam("Elu"));

            std::istringstream ist(cit2->second);
            ist >> param.alpha;

            StaticOp* op = CreateStaticOp(graph, "Elu");
            SetOperatorParam(op, param);
            SetNodeOp(node, op);
            return true;
        }
        else if(cit1->second == "leaky" && cit2 != mxnet_node.attrs.end())
        {
            ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));

            std::istringstream ist(cit2->second);
            ist >> param.negative_slope;

            StaticOp* op = CreateStaticOp(graph, "ReLu");
            SetOperatorParam(op, param);
            SetNodeOp(node, op);
            return true;
        }
    }

    return false;
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
        if(cit->second.find("[")==std::string::npos && cit->second.find(",")==std::string::npos)
        {
            param.re_shape.push_back(atoi(cit->second.c_str()));
        }
        else
        {
            ParseAttr_n(cit->second, v1);
            for(unsigned int i= 0; i < v1.size();++i)
            {
                param.re_shape.push_back(v1.at(i));
            }
            
        }
        
    }

    cit = mxnet_node.attrs.find("reverse");
    
    if(cit != mxnet_node.attrs.end())
    {
        ParseAttr_n(cit->second, v1);
        param.reverse = v1[0];
    }
    param.is_mxnet = true;
    
    StaticOp* op = CreateStaticOp(graph, "Reshape");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;

}
static bool LoadMxnetPermute(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    PermuteParam param = any_cast<PermuteParam>(OpManager::GetOpDefParam("Permute"));

    const_iterator cit;
    std::vector<int> v1;

    cit = mxnet_node.attrs.find("axes");

    ParseAttr_n(cit->second, v1);

    param.order0 = v1[0];
    param.order1 = v1[1];
    param.order2 = v1[2];
    param.order3 = -2;

    StaticOp* op = CreateStaticOp(graph, "Permute");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetSwapAxis(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    SwapAxisParam param = any_cast<SwapAxisParam>(OpManager::GetOpDefParam("SwapAxis"));

    const_iterator cit;
    cit = mxnet_node.attrs.find("dim1");
    if(cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        ist >> param.dim_0;
    }
    cit = mxnet_node.attrs.find("dim2");
    if(cit != mxnet_node.attrs.end())
    {
        std::istringstream ist(cit->second);
        ist >> param.dim_1;
    }

    StaticOp* op = CreateStaticOp(graph, "SwapAxis");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetAddN(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    AddnParam param = any_cast<AddnParam>(OpManager::GetOpDefParam("Addn"));
    param.axis = 1;

    StaticOp* op = CreateStaticOp(graph, "Addn");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetClip(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    const_iterator cit1, cit2;
    cit1 = mxnet_node.attrs.find("a_max");
    cit2 = mxnet_node.attrs.find("a_min");
    if(cit1 != mxnet_node.attrs.end() && cit1->second == "6" && cit2 != mxnet_node.attrs.end() && cit2->second == "0")
    {
        StaticOp* op = CreateStaticOp(graph, "ReLu6");
        SetNodeOp(node, op);
    }
    else
        return false;

    return true;
}

static bool LoadMxnetRNN(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    const_iterator cit = mxnet_node.attrs.find("mode");
    const_iterator cit1 = mxnet_node.attrs.find("state_size");
    int s_size = atoi(cit1->second.c_str());

    if(cit->second == "lstm")
    {
        LSTMParam param = any_cast<LSTMParam>(OpManager::GetOpDefParam("LSTM"));
        param.mxnet_flag = 1;

        param.hidden_size = s_size;
        param.cell_size = s_size;

        StaticOp* op = CreateStaticOp(graph, "LSTM");
        SetOperatorParam(op, param);
        // SetOperatorDynamicShape(op);
        SetNodeOp(node, op);
    }
    else if(cit->second == "gru")
    {
        GRUParam param = any_cast<GRUParam>(OpManager::GetOpDefParam("GRU"));
        param.mxnet_flag = 1;
        param.hidden_size = s_size;

        StaticOp* op = CreateStaticOp(graph, "GRU");
        SetOperatorParam(op, param);
        // SetOperatorDynamicShape(op);
        SetNodeOp(node, op);
    }
    return true;
}
static bool LoadMxnetUpSampling(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    const_iterator cit1, cit2;
    cit1 = mxnet_node.attrs.find("scale");
    cit2 = mxnet_node.attrs.find("sample_type");

    if(cit2 != mxnet_node.attrs.end())
    {
        if(cit2->second == "nearest" || cit2->second == "bilinear")
        {
            InterpParam param = any_cast<InterpParam>(OpManager::GetOpDefParam("Interp"));

            std::istringstream ist(cit1->second);
            ist >> param.height_scale;
            param.width_scale = param.height_scale;
            if(cit2->second == "nearest")
                param.resize_type = 1;
            else
                param.resize_type = 2;

            StaticOp* op = CreateStaticOp(graph, "Interp");
            SetOperatorParam(op, param);
            SetNodeOp(node, op);
        }
    }
    else
    {
        UpsampleParam param = any_cast<UpsampleParam>(OpManager::GetOpDefParam("Upsample"));
        param.scale = atoi(cit1->second.c_str());
        StaticOp* op = CreateStaticOp(graph, "Upsample");
        SetOperatorParam(op, param);
        SetNodeOp(node, op);
    }
    return true;
}
static bool LoadMxnetFlatten(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    FlattenParam param = any_cast<FlattenParam>(OpManager::GetOpDefParam("Flatten"));
    StaticOp* op = CreateStaticOp(graph, "Flatten");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadMxnetCrop(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    CropParam param = any_cast<CropParam>(OpManager::GetOpDefParam("Crop"));
    const_iterator cit1, cit2, cit3;
    cit1 = mxnet_node.attrs.find("h_w");
    cit2 = mxnet_node.attrs.find("offset");
    cit3 = mxnet_node.attrs.find("num_args");
    param.num_args = atoi(cit3->second.c_str());
    param.flag = 1;
    std::vector<int> v1;
    std::vector<int> v2;
    if(cit1 != mxnet_node.attrs.end())
    {
        ParseAttr_n(cit1->second, v1);
        if(v1.size() == 1)
        {
            param.crop_h = v1.at(0);
            param.crop_w = v1.at(0);
        }
        if(v1.size() == 2)
        {
            param.crop_h = v1.at(0);
            param.crop_w = v1.at(1);
        }
    }
    if(cit2 != mxnet_node.attrs.end())
    {
        ParseAttr_n(cit2->second, v2);
        param.offset_h = v2.at(0);
        param.offset_w = v2.at(0);
        param.offset_c = 0;
    }
    StaticOp* op = CreateStaticOp(graph, "Crop");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetEmbed(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    EmbedParam param = any_cast<EmbedParam>(OpManager::GetOpDefParam("Embedding"));
    const_iterator cit1 = mxnet_node.attrs.find("input_dim");
    const_iterator cit2 = mxnet_node.attrs.find("output_dim");
    param.input_dim = atoi(cit1->second.c_str());
    ;
    param.num_output = atoi(cit2->second.c_str());
    ;
    param.weight_data_size = param.input_dim * param.num_output;
#ifdef DEBUG
    std::cout << "EmbedParam : " << param.input_dim << std::endl;
    std::cout << "EmbedParam : " << param.num_output << std::endl;
#endif

    StaticOp* op = CreateStaticOp(graph, "Embedding");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

static bool LoadMxnetInstancenorm(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    InstanceNormParam param = any_cast<InstanceNormParam>(OpManager::GetOpDefParam("InstanceNorm"));
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
        param.eps = 1e-3f;
    }
#ifdef DEBUG
    std::cout << "InstanceNormParam : " << param.eps << std::endl;
#endif
    StaticOp* op = CreateStaticOp(graph, "InstanceNorm");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadMxnetReduction(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    return true;
}
static bool LoadMxnetPsroipooling(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    PsroipoolingParam param = any_cast<PsroipoolingParam>(OpManager::GetOpDefParam("Psroipooling"));
    const_iterator cit1, cit2, cit3;
    cit1 = mxnet_node.attrs.find("pooled_size");
    cit2 = mxnet_node.attrs.find("spatial_scale");
    cit3 = mxnet_node.attrs.find("output_dim");

    if(cit1 != mxnet_node.attrs.end())
    {
        param.pooled_w = atoi(cit1->second.c_str());
        param.pooled_h = atoi(cit1->second.c_str());
    }
    if(cit2 != mxnet_node.attrs.end())
    {
        param.spatial_scale = atoi(cit2->second.c_str());
    }
    else
    {
        param.spatial_scale = 1;
    }

    if(cit3 != mxnet_node.attrs.end())
    {
        param.spatial_scale = atoi(cit3->second.c_str());
    }
    else
    {
        param.output_dim = 1;
    }

    StaticOp* op = CreateStaticOp(graph, "Psroipooling");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
static bool LoadMxnetRoialign(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    RoialignParam param = any_cast<RoialignParam>(OpManager::GetOpDefParam("Roialign"));
    const_iterator cit1, cit2;
    cit1 = mxnet_node.attrs.find("pooled_size");
    cit2 = mxnet_node.attrs.find("spatial_scale");

    std::vector<int> v1;

    if(cit1 != mxnet_node.attrs.end())
    {
        printf("Into sreialize\n");
        ParseAttr_n(cit1->second, v1);
        if(v1.size() == 1)
        {
            param.pooled_width = v1.at(0);
            param.pooled_height = v1.at(0);
        }
        if(v1.size() == 2)
        {
            param.pooled_width = v1.at(0);
            param.pooled_height = v1.at(1);
        }
    }
    if(cit2 != mxnet_node.attrs.end())
    {
        param.spatial_scale = atoi(cit2->second.c_str());
    }
    else
    {
        param.spatial_scale = 1;
    }
    StaticOp* op = CreateStaticOp(graph, "Roialign");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}
static bool LoadMxnetUnaryAbs(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = UNARY_ABS;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
static bool LoadMxnetUnaryNeg(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = UNARY_NEG;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
static bool LoadMxnetUnaryCeil(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = UNARY_CEIL;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
static bool LoadMxnetUnaryFloor(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = UNARY_FLOOR;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
static bool LoadMxnetUnaryCos(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = UNARY_COS;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
static bool LoadMxnetUnarySin(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));

    param.type = UNARY_SIN;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
static bool LoadMxnetUnaryTan(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = UNARY_TAN;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
static bool LoadMxnetUnaryReciprocal(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = UNARY_RECIPROCAL;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
static bool LoadMxnetUnaryAtan(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam("Unary"));
    param.type = UNARY_ATAN;
    StaticOp* op = CreateStaticOp(graph, "Unary");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
static bool LoadMxnetBroadMul(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node)
{
    StaticOp* op = CreateStaticOp(graph, "BroadMul");
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
    p_mxnet->RegisterOpLoadMethod("Deconvolution", op_load_t(LoadMxnetDeConvolution));
    p_mxnet->RegisterOpLoadMethod("Pooling", op_load_t(LoadMxnetPooling));
    p_mxnet->RegisterOpLoadMethod("SoftmaxOutput", op_load_t(LoadMxnetSoftmax));
    p_mxnet->RegisterOpLoadMethod("SoftmaxActivation", op_load_t(LoadMxnetSoftmax));
    p_mxnet->RegisterOpLoadMethod("Concat", op_load_t(LoadMxnetConcat));
    p_mxnet->RegisterOpLoadMethod("BatchNorm", op_load_t(LoadMxnetBatchNorm));
    p_mxnet->RegisterOpLoadMethod("Dropout", op_load_t(LoadMxnetDropout));
    p_mxnet->RegisterOpLoadMethod("Activation", op_load_t(LoadMxnetActivation));

    p_mxnet->RegisterOpLoadMethod("_minus_scalar", op_load_t(LoadMxnetEltScalar));
    p_mxnet->RegisterOpLoadMethod("_mul_scalar", op_load_t(LoadMxnetEltScalar));
    p_mxnet->RegisterOpLoadMethod("elemwise_add", op_load_t(LoadMxnetElemwiseAdd));
    p_mxnet->RegisterOpLoadMethod("LeakyReLU", op_load_t(LoadMxnetLeakyReLU));
    p_mxnet->RegisterOpLoadMethod("FullyConnected", op_load_t(LoadMxnetFullyConnected));

    p_mxnet->RegisterOpLoadMethod("Reshape", op_load_t(LoadMxnetReshape));
    p_mxnet->RegisterOpLoadMethod("SwapAxis", op_load_t(LoadMxnetSwapAxis));
    p_mxnet->RegisterOpLoadMethod("add_n", op_load_t(LoadMxnetAddN));
    p_mxnet->RegisterOpLoadMethod("clip", op_load_t(LoadMxnetClip));
    p_mxnet->RegisterOpLoadMethod("RNN", op_load_t(LoadMxnetRNN));
    p_mxnet->RegisterOpLoadMethod("transpose", op_load_t(LoadMxnetPermute));
    p_mxnet->RegisterOpLoadMethod("UpSampling", op_load_t(LoadMxnetUpSampling));
    p_mxnet->RegisterOpLoadMethod("Crop", op_load_t(LoadMxnetCrop));
    p_mxnet->RegisterOpLoadMethod("Copy", op_load_t(LoadMxnetDropout));

    p_mxnet->RegisterOpLoadMethod("UpSampling", op_load_t(LoadMxnetUpSampling));
    p_mxnet->RegisterOpLoadMethod("Crop", op_load_t(LoadMxnetCrop));
    p_mxnet->RegisterOpLoadMethod("Copy", op_load_t(LoadMxnetDropout));
    p_mxnet->RegisterOpLoadMethod("Flatten", op_load_t(LoadMxnetFlatten));

    p_mxnet->RegisterOpLoadMethod("Embedding", op_load_t(LoadMxnetEmbed));
    p_mxnet->RegisterOpLoadMethod("InstanceNorm", op_load_t(LoadMxnetInstancenorm));
    p_mxnet->RegisterOpLoadMethod("Reduction", op_load_t(LoadMxnetReduction));
    p_mxnet->RegisterOpLoadMethod("_contrib_PSROIPooling", op_load_t(LoadMxnetPsroipooling));
    p_mxnet->RegisterOpLoadMethod("_contrib_ROIAlign", op_load_t(LoadMxnetRoialign));

    p_mxnet->RegisterOpLoadMethod("abs", op_load_t(LoadMxnetUnaryAbs));
    p_mxnet->RegisterOpLoadMethod("neg", op_load_t(LoadMxnetUnaryNeg));
    p_mxnet->RegisterOpLoadMethod("ceil", op_load_t(LoadMxnetUnaryCeil));
    p_mxnet->RegisterOpLoadMethod("floor", op_load_t(LoadMxnetUnaryFloor));
    p_mxnet->RegisterOpLoadMethod("sin", op_load_t(LoadMxnetUnarySin));
    p_mxnet->RegisterOpLoadMethod("cos", op_load_t(LoadMxnetUnaryCos));
    p_mxnet->RegisterOpLoadMethod("atan", op_load_t(LoadMxnetUnaryAtan));
    p_mxnet->RegisterOpLoadMethod("reciprocal", op_load_t(LoadMxnetUnaryReciprocal));
    p_mxnet->RegisterOpLoadMethod("tan", op_load_t(LoadMxnetUnaryTan));
    p_mxnet->RegisterOpLoadMethod("broadcast_mul", op_load_t(LoadMxnetBroadMul));
    p_mxnet->RegisterOpLoadMethod("_div_scalar", op_load_t(LoadMxnetEltScalar));
    p_mxnet->RegisterOpLoadMethod("_plus_scalar", op_load_t(LoadMxnetEltScalar));
    p_mxnet->RegisterOpLoadMethod("elemwise_mul", op_load_t(LoadMxnetElemwiseMul));


    return true;
}

}    // namespace TEngine
