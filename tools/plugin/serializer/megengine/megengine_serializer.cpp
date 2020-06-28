#include <string>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <numeric>

#include "tengine_c_api.h"
#include "data_type.hpp"
#include "exec_attr.hpp"

#include "megengine_serializer.hpp"
#include "megbrain/plugin/opr_footprint.h"
#include "megbrain/utils/json.h"

#include "logger.hpp"
#include "operator_manager.hpp"
#include "operator/conv_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/batch_norm_param.hpp"
#include "operator/gemm_param.hpp"
#include "operator/fc_param.hpp"
#include "operator/eltwise_param.hpp"
#include "operator/reshape_param.hpp"
#include "operator/slice_param.hpp"
#include "operator/squeeze_param.hpp"
#include "operator/relu_param.hpp"
#include "operator/deconv_param.hpp"
#include "operator/resize_param.hpp"
#include "operator/split_param.hpp"
#include "operator/reduction_param.hpp"
#include "operator/transpose_param.hpp"
#include "operator/squeeze_param.hpp"
#include "operator/expanddims_param.hpp"

static const std::unique_ptr<mgb::OprFootprint> opr_footprint_ptr{std::make_unique<mgb::OprFootprint>()};
static mgb::cg::ComputingGraphImpl* cg;

namespace TEngine {

using op_load_t = std::function<bool(StaticGraph*, StaticNode*, const mgb::cg::OperatorNodeBase&)>;

bool MegengineSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    if(file_list.size() != GetFileNum())
        return false;

    std::unique_ptr<mgb::serialization::InputFile> inp_file =
        mgb::serialization::InputFile::make_fs(file_list[0].c_str());
    auto loader = mgb::serialization::GraphLoader::make(std::move(inp_file));
    mgb::serialization::GraphLoadConfig config;
    mgb::serialization::GraphLoader::LoadResult network = loader->load(config, false);

    cg = static_cast<mgb::cg::ComputingGraphImpl*>(network.graph.get());

    SetGraphSource(graph, file_list[0]);
    SetGraphSourceFormat(graph, "megengine");
    SetGraphLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelFormat(graph, MODEL_FORMAT_MEGENGINE);

    return LoadGraph(cg, graph);
}

static bool IteratePruning(mgb::cg::OperatorNodeBase* cur_opr, mgb::cg::OperatorNodeBase* post_opr,
                           std::unordered_set<int>& marked_op)
{
    bool useless = true;
    for(auto& oup : cur_opr->output())
        if(oup->name().find("workspace") == std::string::npos)
        {
            size_t next_checked = 0;
            for(auto& next_op : cg->var_receiver(oup))
            {
                if(marked_op.find(next_op->id()) != marked_op.end() ||
                   (std::strcmp(next_op->dyn_typeinfo()->name, "Reshape") == 0 && oup == next_op->input()[1]))
                    next_checked++;
            }

            useless &= (cg->var_receiver(oup).size() == next_checked);
        }

    if(useless)
    {
        marked_op.insert(cur_opr->id());
        for(auto& inp : cur_opr->input())
        {
            IteratePruning(inp->owner_opr(), cur_opr, marked_op);
        }
    }

    return true;
}

static bool OptimizeForInference(std::vector<std::unique_ptr<mgb::cg::OperatorNodeBase>>& mge_oprs,
                                 std::vector<mgb::cg::OperatorNodeBase*>& ret)
{
    std::vector<mgb::cg::OperatorNodeBase*> reshape_oprs;
    for(auto& opr : mge_oprs)
        if(std::strcmp(opr.get()->dyn_typeinfo()->name, "Reshape") == 0)
        {
            reshape_oprs.push_back(opr.get());
        }

    std::unordered_set<int> marked_op;
    for(auto& reshape_op : reshape_oprs)
    {
        IteratePruning((reshape_op->input()[1])->owner_opr(), reshape_op, marked_op);
    }

    for(auto& opr : mge_oprs)
        if(marked_op.find(opr.get()->id()) == marked_op.end())
        {
            ret.push_back(opr.get());
        }

    return true;
}

static mgb::cg::OperatorNodeBase* subtensor_op = nullptr;

bool MegengineSerializer::LoadGraph(mgb::cg::ComputingGraphImpl* cg, StaticGraph* graph)
{
    SetGraphIdentity(graph, "megengine", "mge-model", "0");

    auto raw_oprs = cg->all_oprs();
    std::vector<mgb::cg::OperatorNodeBase*> mge_oprs;
    OptimizeForInference(raw_oprs, mge_oprs);

    // process op
    for(auto& mge_op : mge_oprs)
    {
        const std::string& mge_op_name = mge_op->name();
        const std::string& mge_op_type = mge_op->dyn_typeinfo()->name;

        // ignore useless inputs
        if(mge_op_type.compare("Host2DeviceCopy") == 0 && (cg->var_receiver(mge_op->output()[0])).size() == 0)
        {
            continue;
        }

        // ignore constant value, like 1, -1
        if(mge_op_type.compare("ImmutableTensor") == 0)
        {
            continue;
        }

        // ignore first subtensor, two subtensor ops as single split op
        if(mge_op_type.compare("Subtensor") == 0 && subtensor_op == nullptr)
        {
            subtensor_op = mge_op;
            continue;
        }

        // create static node in tengine
        StaticNode* node = CreateStaticNode(graph, mge_op_name);
        // process inputs
        if(!LoadNode(graph, node, mge_op))
        {
            return false;
        }

        // process outputs, handle params
        op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(mge_op_type));
        if(!op_func(graph, node, *mge_op))
        {
            return false;
        }
    }

    // add final op to make output tensor 4-dim
    {
        ReshapeParam param = any_cast<ReshapeParam>(OpManager::GetOpDefParam("Reshape"));

        auto oup = (mge_oprs.back())->output()[0];
        auto& shape_vec = oup->shape();
        param.re_shape.push_back(1);
        param.re_shape.push_back(shape_vec[1]);
        param.re_shape.push_back(1);
        param.re_shape.push_back(1);

        StaticNode* node = CreateStaticNode(graph, "output_node");
        StaticTensor* inp_tensor = FindTensor(graph, oup->name());
        AddNodeInputTensor(node, inp_tensor);
        StaticTensor* out_tensor = CreateStaticTensor(graph, "output");
        SetTensorDataType(out_tensor, DataType::GetTypeID("float32"));
        AddNodeOutputTensor(node, out_tensor);
        StaticOp* op = CreateStaticOp(graph, "Reshape");
        SetOperatorParam(op, param);
        SetNodeOp(node, op);
    }

    return true;
}

// forward declaration
static bool LoadMGEConstTensor(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op,
                               int dtype);

// process inputs of each op, leave outputs to every op_func itself
bool MegengineSerializer::LoadNode(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase* mge_op)
{
    const std::string& mge_op_type = mge_op->dyn_typeinfo()->name;
    // ignore shapeof, subtensor, and ConvolutionBackward
    if(mge_op_type.compare("Subtensor") != 0 && mge_op_type.compare("Reshape") != 0 &&
       mge_op_type.compare("ConvolutionBackwardData") != 0)
    {
        auto inputs = mge_op->input();
        for(size_t i = 0; i < inputs.size(); i++)
        {
            const std::string& input_name = inputs[i]->name();
            StaticTensor* tensor = FindTensor(graph, input_name);
            if(tensor == nullptr)
            {
                // process parameter tensor
                StaticNode* const_node = CreateStaticNode(graph, input_name);
                LoadMGEConstTensor(graph, const_node, *(inputs[i]->owner_opr()), DataType::GetTypeID("int"));
            }
            AddNodeInputTensor(node, tensor);
        }
    }
    return true;
}

// get const/param tensor's value
template <typename T> static bool GetConstTensorValue(mgb::cg::VarNode* var_node, T* data, int sz)
{
    mgb::HostTensorND host_val;
    auto func = cg->compile({{var_node, [&](mgb::DeviceTensorND& z) { host_val.copy_from(z).sync(); }}});

    func->execute();
    T* ptr = host_val.ptr<T>();
    for(int i = 0; i < sz; i++)
    {
        data[i] = ptr[i];
    }
    return true;
}

static bool LoadMGEInput(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op)
{
    auto oup = mge_op.output()[0];
    StaticTensor* tensor = CreateStaticTensor(graph, oup->name());
    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    AddNodeOutputTensor(node, tensor);

    std::vector<int> dim;
    for(size_t i = 0; i < (oup->shape()).ndim; i++)
    {
        dim.push_back((oup->shape())[i]);
    }

    SetTensorDim(tensor, dim);
    AddGraphInputNode(graph, node);
    StaticOp* te_op = CreateStaticOp(graph, "InputOp");
    SetNodeOp(node, te_op);
    return true;
}

// in megengine, there are two types of const tensor, valued ones and parameters
static bool LoadMGEConstTensor(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op, int dtype)
{
    auto oup = mge_op.output()[0];
    StaticTensor* tensor = CreateStaticConstTensor(graph, oup->name());

    std::vector<int> dim;
    auto& shape_vec = oup->shape();
    int tensor_size = shape_vec.total_nr_elems();
    for(size_t i = 0; i < oup->shape().ndim; ++i)
    {
        dim.push_back(shape_vec[i]);
    }
    SetTensorDim(tensor, dim);

    SetTensorDataType(tensor, dtype);
    if(dtype == DataType::GetTypeID("float32"))
    {
        int mem_size = sizeof(float) * tensor_size;
        SetTensorSize(tensor, mem_size);

        float* mem_buf = ( float* )std::malloc(mem_size);
        GetConstTensorValue<float>(oup, mem_buf, tensor_size);
        SetConstTensorBuffer(tensor, mem_buf);
    }
    else
    {
        int mem_size = sizeof(int32_t) * tensor_size;
        SetTensorSize(tensor, mem_size);

        int32_t* mem_buf = ( int32_t* )std::malloc(mem_size);
        GetConstTensorValue<int32_t>(oup, mem_buf, tensor_size);
        SetConstTensorBuffer(tensor, mem_buf);
    }

    SetConstTensorFileLocation(tensor, -1, 0);
    StaticOp* te_op = CreateStaticOp(graph, "Const");
    SetNodeOp(node, te_op);
    AddNodeOutputTensor(node, tensor);
    return true;
}

static bool LoadMGEParamTensor(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op)
{
    return LoadMGEConstTensor(graph, node, mge_op, DataType::GetTypeID("float32"));
}

static const std::unordered_map<std::string, std::string> activation_type = {
    {"RELU", "ReLu"},
    {"TANH", "Tanh"},
    {"SIGMOID", "Sigmoid"},
};

static const std::unordered_map<std::string, EltType> eltwise_type = {
    {"ADD", ELT_SUM},      {"SUB", ELT_SUB},      {"MUL", ELT_PROD},     {"TRUE_DIV", ELT_DIV},
    {"TRUE_LOG", ELT_LOG}, {"TRUE_EXP", ELT_EXP}, {"TRUE_MAX", ELT_MAX}, {"EXP", ELT_EXP},
};

static bool LoadMGEEltwise(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op)
{
    // get mode for json
    auto mge_params = (opr_footprint_ptr->calc_footprint(const_cast<mgb::cg::OperatorNodeBase*>(&mge_op))).param;
    auto mge_param_obj = *(static_cast<mgb::json::Object*>(mge_params.get()));
    const std::string& mode = (static_cast<mgb::json::String*>(mge_param_obj["mode"].get()))->get_impl();

    auto oup = mge_op.output()[0];
    const std::string& output_name = oup->name();
    StaticTensor* tensor = CreateStaticTensor(graph, output_name);
    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    AddNodeOutputTensor(node, tensor);

    StaticOp* te_op;
    if(activation_type.find(mode) != activation_type.end())
    {
        // activation
        te_op = CreateStaticOp(graph, activation_type.at(mode));
        if(mode.compare("RELU") == 0)
        {
            ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));
            param.negative_slope = 0.f;
            SetOperatorParam(te_op, param);
        }
    }
    else
    {
        // math elemwise
        EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));

        param.type = eltwise_type.at(mode);
        param.shift = 1.0;
        param.scale = 1.0;
        param.power = 0.0;

        te_op = CreateStaticOp(graph, "Eltwise");
        SetOperatorParam(te_op, param);
    }

    SetNodeOp(node, te_op);
    return true;
}

static bool LoadMGEPooling(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op)
{
    auto oup = mge_op.output()[0];
    const std::string& output_name = oup->name();
    StaticTensor* tensor = CreateStaticTensor(graph, output_name);
    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    AddNodeOutputTensor(node, tensor);

    auto mge_params = (opr_footprint_ptr->calc_footprint(const_cast<mgb::cg::OperatorNodeBase*>(&mge_op))).param;
    auto mge_param_obj = *(static_cast<mgb::json::Object*>(mge_params.get()));
    int kernel_w = (static_cast<mgb::json::NumberInt*>(mge_param_obj["window_w"].get()))->get_impl();
    int kernel_h = (static_cast<mgb::json::NumberInt*>(mge_param_obj["window_h"].get()))->get_impl();
    int stride_w = (static_cast<mgb::json::NumberInt*>(mge_param_obj["stride_w"].get()))->get_impl();
    int stride_h = (static_cast<mgb::json::NumberInt*>(mge_param_obj["stride_h"].get()))->get_impl();
    int pad_w = (static_cast<mgb::json::NumberInt*>(mge_param_obj["pad_w"].get()))->get_impl();
    int pad_h = (static_cast<mgb::json::NumberInt*>(mge_param_obj["pad_h"].get()))->get_impl();
    const std::string& alg = (static_cast<mgb::json::String*>(mge_param_obj["mode"].get()))->get_impl();

    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));
    param.kernel_w = kernel_w;
    param.kernel_h = kernel_h;
    param.stride_w = stride_w;
    param.stride_h = stride_h;
    param.pad_w0 = pad_w;
    param.pad_w1 = pad_w;
    param.pad_h0 = pad_h;
    param.pad_h1 = pad_h;

    if(alg.compare("AVERAGE") == 0)
    {
        param.alg = kPoolAvg;
    }
    else
    {
        param.alg = kPoolMax;
    }

    StaticOp* te_op = CreateStaticOp(graph, "Pooling");
    SetOperatorParam(te_op, param);
    SetNodeOp(node, te_op);
    return true;
}

static bool LoadMGEConvolution(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op)
{
    auto oup = mge_op.output()[0];
    const std::string& output_name = oup->name();
    StaticTensor* tensor = CreateStaticTensor(graph, output_name);
    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    AddNodeOutputTensor(node, tensor);

    auto mge_params = (opr_footprint_ptr->calc_footprint(const_cast<mgb::cg::OperatorNodeBase*>(&mge_op))).param;
    auto mge_param_obj = *(static_cast<mgb::json::Object*>(mge_params.get()));
    int stride_w = (static_cast<mgb::json::NumberInt*>(mge_param_obj["stride_w"].get()))->get_impl();
    int stride_h = (static_cast<mgb::json::NumberInt*>(mge_param_obj["stride_h"].get()))->get_impl();
    int pad_w = (static_cast<mgb::json::NumberInt*>(mge_param_obj["pad_w"].get()))->get_impl();
    int pad_h = (static_cast<mgb::json::NumberInt*>(mge_param_obj["pad_h"].get()))->get_impl();
    int dilate_w = (static_cast<mgb::json::NumberInt*>(mge_param_obj["dilate_w"].get()))->get_impl();
    int dilate_h = (static_cast<mgb::json::NumberInt*>(mge_param_obj["dilate_h"].get()))->get_impl();

    const std::string& sparse = (static_cast<mgb::json::String*>(mge_param_obj["sparse"].get()))->get_impl();

    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));
    param.kernel_w = ((mge_op.input())[1]->shape())[3];
    param.kernel_h = ((mge_op.input())[1]->shape())[2];
    param.stride_w = stride_w;
    param.stride_h = stride_h;
    param.pad_w0 = pad_w;
    param.pad_w1 = pad_w;
    param.pad_h0 = pad_h;
    param.pad_h1 = pad_h;
    param.dilation_w = dilate_w;
    param.dilation_h = dilate_h;

    param.group = 1;
    param.output_channel = ((mge_op.output())[0]->shape())[1];

    if(sparse.compare("GROUP") == 0)
    {
        param.kernel_w = ((mge_op.input())[1]->shape())[4];
        param.kernel_h = ((mge_op.input())[1]->shape())[3];

        param.group = ((mge_op.input())[1]->shape())[0];
        param.output_channel = ((mge_op.input())[1]->shape())[1] * param.group;

        std::vector<int> dims;

        dims.push_back(param.output_channel);
        dims.push_back(1);
        dims.push_back(param.kernel_h);
        dims.push_back(param.kernel_w);

        StaticTensor* weight_tensor = FindTensor(graph, mge_op.input()[1]->name());
        SetTensorDim(weight_tensor, dims);
    }

    StaticOp* te_op = CreateStaticOp(graph, "Convolution");
    SetOperatorParam(te_op, param);
    SetNodeOp(node, te_op);
    return true;
}

static bool LoadMGEReshape(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op)
{
    StaticTensor* input_tensor = FindTensor(graph, mge_op.input()[0]->name());
    AddNodeInputTensor(node, input_tensor);

    ReshapeParam param = any_cast<ReshapeParam>(OpManager::GetOpDefParam("Reshape"));

    auto oup = mge_op.output()[0];
    auto& shape_vec = oup->shape();
    for(size_t i = 0; i < shape_vec.ndim; ++i)
    {
        param.re_shape.push_back(shape_vec[i]);
    }

    StaticTensor* tensor = CreateStaticTensor(graph, oup->name());
    SetTensorDataType(tensor, input_tensor->data_type);
    AddNodeOutputTensor(node, tensor);

    StaticOp* op = CreateStaticOp(graph, "Reshape");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadMGEConcat(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op)
{
    auto oup = mge_op.output()[0];
    const std::string& output_name = oup->name();
    StaticTensor* tensor = CreateStaticTensor(graph, output_name);
    StaticTensor* source_tensor = FindTensor(graph, (mge_op.input()[0])->name());
    SetTensorDataType(tensor, source_tensor->data_type);
    AddNodeOutputTensor(node, tensor);

    auto mge_params = (opr_footprint_ptr->calc_footprint(const_cast<mgb::cg::OperatorNodeBase*>(&mge_op))).param;
    auto mge_param_obj = *(static_cast<mgb::json::Object*>(mge_params.get()));
    int axis = (static_cast<mgb::json::NumberInt*>(mge_param_obj["axis"].get()))->get_impl();

    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));
    param.axis = axis;

    StaticOp* op = CreateStaticOp(graph, "Concat");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadMGEBatchNorm(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op)
{
    auto oup = mge_op.output()[4];
    const std::string& output_name = oup->name();
    StaticTensor* tensor = CreateStaticTensor(graph, output_name);
    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    AddNodeOutputTensor(node, tensor);

    BatchNormParam param = any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));
    param.eps = 1e-5;

    StaticOp* op = CreateStaticOp(graph, "BatchNormalization");
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

static bool LoadMGEIdentity(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op)
{
    auto oup = mge_op.output()[0];
    const std::string& output_name = oup->name();
    StaticTensor* tensor = CreateStaticTensor(graph, output_name);
    StaticTensor* source_tensor = FindTensor(graph, (mge_op.input()[0])->name());
    SetTensorDataType(tensor, source_tensor->data_type);
    AddNodeOutputTensor(node, tensor);

    StaticOp* te_op = CreateStaticOp(graph, "Noop");
    SetNodeOp(node, te_op);
    return true;
}

static bool LoadMGEMatrixMul(StaticGraph* graph, StaticNode* node, const mgb::cg::OperatorNodeBase& mge_op)
{
    auto oup = mge_op.output()[0];
    const std::string& output_name = oup->name();
    StaticTensor* tensor = CreateStaticTensor(graph, output_name);
    SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    AddNodeOutputTensor(node, tensor);

    auto mge_params = (opr_footprint_ptr->calc_footprint(const_cast<mgb::cg::OperatorNodeBase*>(&mge_op))).param;
    auto mge_param_obj = *(static_cast<mgb::json::Object*>(mge_params.get()));
    bool trans_A = (static_cast<mgb::json::Bool*>(mge_param_obj["transposeA"].get()))->get_impl();
    bool trans_B = (static_cast<mgb::json::Bool*>(mge_param_obj["transposeB"].get()))->get_impl();

    GemmParam param = any_cast<GemmParam>(OpManager::GetOpDefParam("Gemm"));
    param.transA = trans_A;
    param.transB = trans_B;
    param.alpha = 1;
    param.beta = 1;

    if(param.transA)
    {
        StaticOp* te_op = CreateStaticOp(graph, "Gemm");
        SetOperatorParam(te_op, param);
        SetNodeOp(node, te_op);
        return true;
    }

    StaticTensor* weight_tensor = FindTensor(graph, (mge_op.input()[1])->name());
    if(!param.transB)
    {
        int k = weight_tensor->dims[0], n = weight_tensor->dims[1];
        weight_tensor->dims[0] = n;
        weight_tensor->dims[1] = k;

        float* tmp = ( float* )malloc(k * n * sizeof(float));
        // if second input is not const, this would not work
        float* data = ( float* )GetConstTensorBuffer(weight_tensor);

        for(int i = 0; i < n; i++)
            for(int j = 0; j < k; j++)
            {
                tmp[i * k + j] = data[j * n + i];
            }

        memcpy(data, tmp, n * k * sizeof(float));

        free(tmp);
    }

    StaticOp* op = CreateStaticOp(graph, "FullyConnected");

    FCParam fc_param = any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));
    fc_param.num_output = weight_tensor->dims[0];

    SetOperatorParam(op, fc_param);
    SetNodeOp(node, op);
    return true;
}

bool MegengineSerializerRegisterOpLoader(void)
{
    SerializerPtr serializer;

    if(!SerializerManager::SafeGet("megengine", serializer))
        return false;

    MegengineSerializer* p_mge = dynamic_cast<MegengineSerializer*>(serializer.get());

    p_mge->RegisterOpLoadMethod("Host2DeviceCopy", op_load_t(LoadMGEInput));
    p_mge->RegisterOpLoadMethod("SharedDeviceTensor", op_load_t(LoadMGEParamTensor));
    p_mge->RegisterOpLoadMethod("Elemwise", op_load_t(LoadMGEEltwise));
    p_mge->RegisterOpLoadMethod("PoolingForward", op_load_t(LoadMGEPooling));
    p_mge->RegisterOpLoadMethod("ConvolutionForward", op_load_t(LoadMGEConvolution));
    p_mge->RegisterOpLoadMethod("Reshape", op_load_t(LoadMGEReshape));
    p_mge->RegisterOpLoadMethod("Concat", op_load_t(LoadMGEConcat));
    p_mge->RegisterOpLoadMethod("BatchNormForward", op_load_t(LoadMGEBatchNorm));
    p_mge->RegisterOpLoadMethod("Identity", op_load_t(LoadMGEIdentity));
    p_mge->RegisterOpLoadMethod("MatrixMul", op_load_t(LoadMGEMatrixMul));
    p_mge->RegisterOpLoadMethod("MarkNoBroadcastElemwise", op_load_t(LoadMGEIdentity));

    return true;
}

}    // namespace TEngine
