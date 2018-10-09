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
 * Author: haitao@openailab.com
 * Author: chunyinglv@openailab.com
 */
#include <functional>
#include <iostream>
#include <unordered_map>
#include "compiler.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "caffe_serializer.hpp"
#include "operator/batch_norm_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/conv_param.hpp"
#include "operator/deconv_param.hpp"
#include "operator/detection_output_param.hpp"
#include "operator/eltwise_param.hpp"
#include "operator/fc_param.hpp"
#include "operator/flatten_param.hpp"
#include "operator/lrn_param.hpp"
#include "operator/normalize_param.hpp"
#include "operator/permute_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/priorbox_param.hpp"
#include "operator/region_param.hpp"
#include "operator/relu_param.hpp"
#include "operator/reorg_param.hpp"
#include "operator/reshape_param.hpp"
#include "operator/resize_param.hpp"
#include "operator/roi_pooling_param.hpp"
#include "operator/rpn_param.hpp"
#include "operator/scale_param.hpp"
#include "operator/slice_param.hpp"
#include "operator/softmax_param.hpp"
#include "operator_manager.hpp"
#include "type_name.hpp"

namespace TEngine {

using op_load_t = std::function<bool(StaticGraph*, StaticNode*,
                                     const te_caffe::LayerParameter&)>;
using blob_load_t = std::function<bool(StaticGraph*, StaticNode*,
                                       const te_caffe::LayerParameter&)>;

std::unordered_map<std::string, blob_load_t> blob_load_map;

// Check if NetParameter uses old style V0LayerParameter
static bool NetNeedsV0ToV1Upgrade(const te_caffe::NetParameter& caffe_net) {
  for (int i = 0; i < caffe_net.layers_size(); ++i) {
    if (caffe_net.layers(i).has_layer()) return true;
  }
  return false;
}

// Check if NetParameter uses old style V1LayerParameter
static bool NetNeedsV1ToV2Upgrade(const te_caffe::NetParameter& caffe_net) {
  return (caffe_net.layers_size() > 0);
}

// Check if NetParameter uses old style input fields
static bool NetNeedsInputUpgrade(const te_caffe::NetParameter& caffe_net) {
  return (caffe_net.input_size() > 0);
}

// Check if NetParameter uses old style data transformation fields
static bool NetNeedsDataUpgrade(const te_caffe::NetParameter& caffe_net) {
  for (int i = 0; i < caffe_net.layers_size(); ++i) {
    if (caffe_net.layers(i).type() ==
        te_caffe::V1LayerParameter_LayerType_DATA) {
      te_caffe::DataParameter layer_param = caffe_net.layers(i).data_param();
      if (layer_param.has_scale() || layer_param.has_mean_file() ||
          layer_param.has_crop_size() || layer_param.has_mirror())
        return true;
    }
    if (caffe_net.layers(i).type() ==
        te_caffe::V1LayerParameter_LayerType_IMAGE_DATA) {
      te_caffe::ImageDataParameter layer_param =
          caffe_net.layers(i).image_data_param();
      if (layer_param.has_scale() || layer_param.has_mean_file() ||
          layer_param.has_crop_size() || layer_param.has_mirror())
        return true;
    }
    if (caffe_net.layers(i).type() ==
        te_caffe::V1LayerParameter_LayerType_WINDOW_DATA) {
      te_caffe::WindowDataParameter layer_param =
          caffe_net.layers(i).window_data_param();
      if (layer_param.has_scale() || layer_param.has_mean_file() ||
          layer_param.has_crop_size() || layer_param.has_mirror())
        return true;
    }
  }
  return false;
}

static bool NetNeedsUpgrade(const char* fname,
                            const te_caffe::NetParameter& caffe_net) {
  if (NetNeedsV0ToV1Upgrade(caffe_net) || NetNeedsV1ToV2Upgrade(caffe_net) ||
      NetNeedsInputUpgrade(caffe_net) || NetNeedsDataUpgrade(caffe_net)) {
    LOG_ERROR() << "The input file specified is using deprecated params: "
                << fname << "\n";
    LOG_ERROR() << "Please upgrade the input file by using caffe "
                   "tools(upgrade_net_proto_text/upgrade_net_proto_binary).\n";
    return true;
  }
  return false;
}

bool CaffeSingle::LoadBinaryFile(const char* fname,
                                 te_caffe::NetParameter& caffe_net) {
  std::ifstream is(fname, std::ios::in | std::ios::binary);

  if (!is.is_open()) {
    LOG_ERROR() << "cannot open file: " << fname << "\n";
    return false;
  }

  google::protobuf::io::IstreamInputStream input_stream(&is);
  google::protobuf::io::CodedInputStream coded_input(&input_stream);
  // SetTotalBytesLimit(max_limit, warning_threshold)
  coded_input.SetTotalBytesLimit(1024 << 20, 512 << 20);

  bool ret = caffe_net.ParseFromCodedStream(&coded_input);

  is.close();

  if (!ret) LOG_ERROR() << "parse file: " << fname << " failed\n";

  if (NetNeedsUpgrade(fname, caffe_net)) return false;

  return ret;
}

bool CaffeSingle::LoadTextFile(const char* fname,
                               te_caffe::NetParameter& caffe_net) {
  std::ifstream is(fname, std::ios::in);

  if (!is.is_open()) {
    LOG_ERROR() << "cannot open file: " << fname << "\n";
    return false;
  }

  google::protobuf::io::IstreamInputStream input_stream(&is);
  bool ret = google::protobuf::TextFormat::Parse(&input_stream, &caffe_net);

  is.close();

  if (!ret) LOG_ERROR() << "parse file: " << fname << " failed\n";

  if (NetNeedsUpgrade(fname, caffe_net)) return false;

  return ret;
}

bool CaffeSingle::LoadModel(const std::vector<std::string>& file_list,
                            StaticGraph* graph) {
  te_caffe::NetParameter caffe_net;

  if (file_list.size() != GetFileNum()) return false;

  if (!LoadBinaryFile(file_list[0].c_str(), caffe_net)) return false;

  SetGraphSource(graph, file_list[0]);
  SetGraphSourceFormat(graph, "caffe");
  SetGraphConstTensorFile(graph, file_list[0]);

  return LoadGraph(caffe_net, graph);
}

bool CaffeSingle::LoadNode(StaticGraph* graph, StaticNode* node,
                           const te_caffe::LayerParameter& layer_param,
                           name_map_t& tensor_name_map) {
  for (int i = 0; i < layer_param.bottom_size(); i++) {
    const std::string& orig_name = layer_param.bottom(i);

    std::string& tensor_name = tensor_name_map[orig_name];

    StaticTensor* tensor = FindTensor(graph, tensor_name);

    AddNodeInputTensor(node, tensor);
  }

  for (int i = 0; i < layer_param.top_size(); i++) {
    const std::string& orig_name = layer_param.top(i);
    std::string tensor_name;

    if (tensor_name_map.count(orig_name))
      tensor_name = GetNodeName(node) + "/" + std::to_string(i);
    else
      tensor_name = orig_name;

    StaticTensor* tensor = CreateStaticTensor(graph, tensor_name);

    SetTensorDataLayout(tensor, "NCHW");
    SetTensorDataType(tensor, "float32");

    AddNodeOutputTensor(node, tensor);

    // record the name mapping

    tensor_name_map[orig_name] = tensor_name;
  }

  return true;
}

bool CaffeSingle::LoadGraph(te_caffe::NetParameter& caffe_net,
                            StaticGraph* graph) {
  SetGraphIdentity(graph, "caffe", caffe_net.name(), "0");

  name_map_t tensor_name_map;

  int layer_num = caffe_net.layer_size();
  int i;

  for (i = 0; i < layer_num; i++) {
    const te_caffe::LayerParameter& layer_param = caffe_net.layer(i);
    const std::string& caffe_op_name = layer_param.type();

    if (!FindOpLoadMethod(caffe_op_name)) {
      LOG_ERROR() << "cannot find load function for operator: " << caffe_op_name
                  << "\n";
      break;
    }

    StaticNode* node = CreateStaticNode(graph, layer_param.name());

    if (!LoadNode(graph, node, layer_param, tensor_name_map)) break;

    op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(caffe_op_name));

    if (!op_func(graph, node, layer_param)) break;
  }

  if (i < layer_num) return false;

  return true;
}

bool CaffeBuddy::LoadModel(const std::vector<std::string>& file_list,
                           StaticGraph* graph) {
  if (file_list.size() != GetFileNum()) return false;

  te_caffe::NetParameter test_net;

  if (!LoadTextFile(file_list[0].c_str(), test_net)) {
    std::cout << "FAILED\n";
    return false;
  }

  te_caffe::NetParameter train_net;

  if (!LoadBinaryFile(file_list[1].c_str(), train_net)) return false;

  SetGraphSource(graph, file_list[1]);
  SetGraphSourceFormat(graph, "caffe");
  SetGraphConstTensorFile(graph, file_list[1]);

  return LoadGraph(test_net, train_net, graph);
}

bool CaffeBuddy::LoadGraph(te_caffe::NetParameter& test_net,
                           te_caffe::NetParameter& train_net,
                           StaticGraph* graph) {
  name_map_t tensor_name_map;

  SetGraphIdentity(graph, "caffe", test_net.name(), "0");

  /* create the layer name map of the train_net */
  std::unordered_map<std::string, const te_caffe::LayerParameter*>
      train_name_map;

  int layer_number;

  layer_number = train_net.layer_size();

  int i;

  for (i = 0; i < layer_number; i++) {
    const te_caffe::LayerParameter& layer_param = train_net.layer(i);

    train_name_map[layer_param.name()] = &layer_param;
  }

  layer_number = test_net.layer_size();
  int n;

  for (n = 0; n < layer_number; n++) {
    const te_caffe::LayerParameter& layer_param = test_net.layer(n);
    const std::string& caffe_op_name = layer_param.type();

    if (!FindOpLoadMethod(caffe_op_name)) {
      LOG_ERROR() << "cannot find load function for operator: " << caffe_op_name
                  << "\n";
      break;
    }

    StaticNode* node = CreateStaticNode(graph, layer_param.name());

    if (!LoadNode(graph, node, layer_param, tensor_name_map)) break;

    op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(caffe_op_name));

    if (!op_func(graph, node, layer_param)) break;

    /*Load pre-trained parameters*/
    if (train_name_map.count(layer_param.name())) {
      const te_caffe::LayerParameter* p_train;

      p_train = train_name_map[layer_param.name()];

      if (p_train->blobs_size()) {
        blob_load_t func = blob_load_map[caffe_op_name];

        if (!func(graph, node, *p_train)) break;
      }
    }
  }

  if (n < layer_number) return false;

  return true;
}

static void LoadCaffeBlob(StaticGraph* graph, StaticNode* node,
                          const std::vector<std::string>& name_list,
                          const std::vector<std::string>& layout_list,
                          const te_caffe::LayerParameter& layer_param)

{
  unsigned int blob_num = layer_param.blobs_size();

  for (unsigned int i = 0; i < blob_num && i < name_list.size(); i++) {
    std::string new_tensor_name = GetNodeName(node) + "/" + name_list[i];

    StaticTensor* tensor = CreateStaticConstTensor(graph, new_tensor_name);

    /* load tensor data*/

    const te_caffe::BlobProto& blob = layer_param.blobs(i);

    std::vector<int> dims;

    if (blob.has_shape()) {
      for (int i = 0; i < blob.shape().dim_size(); i++) {
        dims.push_back(blob.shape().dim(i));
      }
    } else {
      std::vector<int> temp;
      temp.push_back(blob.num());
      temp.push_back(blob.channels());
      temp.push_back(blob.height());
      temp.push_back(blob.width());

      int start = 0;

      while (temp[start] == 1) start++;

      for (unsigned int i = start; i < temp.size(); i++)
        dims.push_back(temp[i]);
    }

    SetTensorDim(tensor, dims);
    SetTensorDataType(tensor, "float32");
    SetTensorDataLayout(tensor, layout_list[i]);

    int mem_size = blob.data_size() * 4;

    SetTensorSize(tensor, mem_size);

    float* ptr = (float*)std::malloc(mem_size + 128);

    for (int i = 0; i < blob.data_size(); i++) ptr[i] = blob.data(i);

    SetConstTensorBuffer(tensor, ptr);
    SetConstTensorFileLocation(tensor, -1, 0);

    StaticNode* new_node = CreateStaticNode(graph, new_tensor_name);

    StaticOp* const_op = CreateStaticOp(graph, "Const");

    SetNodeOp(new_node, const_op);

    AddNodeOutputTensor(new_node, tensor);

    AddNodeInputTensor(node, tensor);
  }
}

static void CreatePresetNode(StaticGraph* graph, StaticNode* node,
                             const char* name, const char* layout,
                             std::vector<int>& dims, float val) {
  std::string new_tensor_name = GetNodeName(node) + "/" + name;
  StaticTensor* tensor = CreateStaticConstTensor(graph, new_tensor_name);

  SetTensorDim(tensor, dims);
  SetTensorDataType(tensor, "float32");
  SetTensorDataLayout(tensor, layout);

  int elem_size = 1;

  for (unsigned int i = 0; i < dims.size(); i++) {
    elem_size *= dims[i];
  }

  SetTensorSize(tensor, elem_size * sizeof(float));

  float* ptr = (float*)std::malloc(elem_size * sizeof(float));

  for (int i = 0; i < elem_size; i++) ptr[i] = val;

  SetConstTensorBuffer(tensor, ptr);
  SetConstTensorFileLocation(tensor, -1, 0);

  StaticNode* new_node = CreateStaticNode(graph, new_tensor_name);

  StaticOp* const_op = CreateStaticOp(graph, "Const");

  SetNodeOp(new_node, const_op);

  AddNodeOutputTensor(new_node, tensor);

  AddNodeInputTensor(node, tensor);
}

static bool LoadBatchNormBlob(StaticGraph* graph, StaticNode* node,
                              const te_caffe::LayerParameter& layer_param) {
  const te_caffe::BlobProto& rescale_blob = layer_param.blobs(2);

  StaticOp* op = GetNodeOp(node);

  BatchNormParam param = any_cast<BatchNormParam>(GetOperatorParam(op));

  param.rescale_factor = rescale_blob.data(0);

  SetOperatorParam(op, param);

  /* for compatible reason, create the two tensors: gamma (1.0) and beta (0.0)
   */

  /* get the dim, i.e., channel size */

  const te_caffe::BlobProto& mean_blob = layer_param.blobs(0);

  std::vector<int> dims;
  dims.push_back(mean_blob.shape().dim(0));

  CreatePresetNode(graph, node, "gamma", "W", dims, 1.0f);
  CreatePresetNode(graph, node, "beta", "W", dims, 0.0f);

  std::vector<std::string> name_list = {"means", "vars"};
  std::vector<std::string> layout_list = {"W", "W"};

  LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

  return true;
}

static bool LoadFullyConnectedBlob(
    StaticGraph* graph, StaticNode* node,
    const te_caffe::LayerParameter& layer_param) {
  std::vector<std::string> name_list = {"weight", "bias"};
  std::vector<std::string> layout_list = {"HW", "W"};

  LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

  return true;
}

static bool LoadScaleBlob(StaticGraph* graph, StaticNode* node,
                          const te_caffe::LayerParameter& layer_param) {
  std::vector<std::string> name_list = {"gamma", "beta"};
  std::vector<std::string> layout_list = {"CHW", "W"};

  LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

  return true;
}
static bool LoadPReLuBlob(StaticGraph* graph, StaticNode* node,
                          const te_caffe::LayerParameter& layer_param) {
  std::vector<std::string> name_list = {"slope"};
  std::vector<std::string> layout_list = {"W"};
  LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

  return true;
}
static bool LoadNormalizeBlob(StaticGraph* graph, StaticNode* node,
                              const te_caffe::LayerParameter& layer_param) {
  std::vector<std::string> name_list = {"scale"};
  std::vector<std::string> layout_list = {"W"};
  LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

  return true;
}
static bool LoadConvolutionBlob(StaticGraph* graph, StaticNode* node,
                                const te_caffe::LayerParameter& layer_param) {
  std::vector<std::string> name_list = {"weight", "bias"};
  std::vector<std::string> layout_list = {"NCHW", "W"};

  LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

  return true;
}
static bool LoadDeconvolutionBlob(StaticGraph* graph, StaticNode* node,
                                  const te_caffe::LayerParameter& layer_param) {
  std::vector<std::string> name_list = {"weight", "bias"};
  std::vector<std::string> layout_list = {"NCHW", "C"};

  LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

  return true;
}
static bool LoadCaffeInputOp(StaticGraph* graph, StaticNode* node,
                             const te_caffe::LayerParameter& layer_param) {
  StaticOp* op = CreateStaticOp(graph, "InputOp");

  SetNodeOp(node, op);

  const te_caffe::InputParameter& input_param = layer_param.input_param();

  if (input_param.shape_size()) {
    std::vector<int> dim;
    const te_caffe::BlobShape& blob_shape = input_param.shape(0);

    for (int i = 0; i < blob_shape.dim_size(); i++) {
      dim.push_back(blob_shape.dim(i));
    }

    StaticTensor* tensor = GetNodeOutputTensor(graph, node, 0);

    SetTensorDim(tensor, dim);
  }

  AddGraphInputNode(graph, node);

  return true;
}

static bool LoadCaffeSoftmax(StaticGraph* graph, StaticNode* node,
                             const te_caffe::LayerParameter& layer_param) {
  const te_caffe::SoftmaxParameter& softmax_param = layer_param.softmax_param();

  SoftmaxParam param =
      any_cast<SoftmaxParam>(OpManager::GetOpDefParam("Softmax"));

  if (softmax_param.has_axis())
    param.axis = softmax_param.axis();
  else
    param.axis = 1;

  StaticOp* op = CreateStaticOp(graph, "Softmax");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}

static bool LoadCaffeReorg(StaticGraph* graph, StaticNode* node,
                           const te_caffe::LayerParameter& layer_param) {
  const te_caffe::ReorgParameter& caffe_param = layer_param.reorg_param();

  ReorgParam param = any_cast<ReorgParam>(OpManager::GetOpDefParam("Reorg"));

  param.stride = caffe_param.stride();

  StaticOp* op = CreateStaticOp(graph, "Reorg");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}
static bool LoadCaffeRegion(StaticGraph* graph, StaticNode* node,
                            const te_caffe::LayerParameter& layer_param) {
  const te_caffe::RegionParameter& caffe_param = layer_param.region_param();

  RegionParam param = any_cast<RegionParam>(OpManager::GetOpDefParam("Region"));

  param.num_classes = caffe_param.num_classes();
  param.num_box = caffe_param.num_box();
  param.side = caffe_param.side();
  param.coords = caffe_param.coords();
  param.confidence_threshold = caffe_param.confidence_threshold();
  param.nms_threshold = caffe_param.nms_threshold();

  for (int i = 0; i < (int)caffe_param.biases_size(); ++i) {
    param.biases.push_back(caffe_param.biases(i));
  }

  StaticOp* op = CreateStaticOp(graph, "Region");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}

static bool LoadCaffeNormalize(StaticGraph* graph, StaticNode* node,
                               const te_caffe::LayerParameter& layer_param) {
  const te_caffe::NormalizeParameter& normalize_param =
      layer_param.norm_param();

  NormalizeParam param =
      any_cast<NormalizeParam>(OpManager::GetOpDefParam("Normalize"));

  param.across_spatial = normalize_param.across_spatial();
  param.channel_shared = normalize_param.channel_shared();

  StaticOp* op = CreateStaticOp(graph, "Normalize");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}

static bool LoadCaffeSlice(StaticGraph* graph, StaticNode* node,
                           const te_caffe::LayerParameter& layer_param) {
  const te_caffe::SliceParameter& slice_param = layer_param.slice_param();

  SliceParam param = any_cast<SliceParam>(OpManager::GetOpDefParam("Slice"));

  if (slice_param.has_axis())
    param.axis = slice_param.axis();
  else
    param.axis = 1;

  StaticOp* op = CreateStaticOp(graph, "Slice");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}

static bool LoadCaffeReLu(StaticGraph* graph, StaticNode* node,
                          const te_caffe::LayerParameter& layer_param) {
  ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam("ReLu"));

  const te_caffe::ReLUParameter& caffe_param = layer_param.relu_param();

  if (caffe_param.has_negative_slope())
    param.negative_slope = static_cast<float>(caffe_param.negative_slope());
  else
    param.negative_slope = 0.f;

  StaticOp* op = CreateStaticOp(graph, "ReLu");
  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}

static bool LoadCaffeSplit(StaticGraph* graph, StaticNode* node,
                           const te_caffe::LayerParameter& layer_param) {
  StaticOp* op = CreateStaticOp(graph, "Split");

  SetNodeOp(node, op);

  return true;
}

static bool LoadCaffeConcat(StaticGraph* graph, StaticNode* node,
                            const te_caffe::LayerParameter& layer_param) {
  ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam("Concat"));

  const te_caffe::ConcatParameter& concat_param = layer_param.concat_param();

  if (concat_param.has_concat_dim())
    param.axis = static_cast<int>(concat_param.concat_dim());
  else
    param.axis = concat_param.axis();

  StaticOp* op = CreateStaticOp(graph, "Concat");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}
static bool LoadCaffePermute(StaticGraph* graph, StaticNode* node,
                             const te_caffe::LayerParameter& layer_param) {
  PermuteParam param =
      any_cast<PermuteParam>(OpManager::GetOpDefParam("Permute"));

  const te_caffe::PermuteParameter& permute_param = layer_param.permute_param();

  param.order0 = permute_param.order(0);
  param.order1 = permute_param.order(1);
  param.order2 = permute_param.order(2);
  param.order3 = permute_param.order(3);

  StaticOp* op = CreateStaticOp(graph, "Permute");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}
static bool LoadCaffeFlatten(StaticGraph* graph, StaticNode* node,
                             const te_caffe::LayerParameter& layer_param) {
  FlattenParam param =
      any_cast<FlattenParam>(OpManager::GetOpDefParam("Flatten"));

  const te_caffe::FlattenParameter& flatten_param = layer_param.flatten_param();

  param.axis = flatten_param.axis();

  StaticOp* op = CreateStaticOp(graph, "Flatten");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}
static bool LoadCaffePriorBox(StaticGraph* graph, StaticNode* node,
                              const te_caffe::LayerParameter& layer_param) {
  PriorBoxParam param =
      any_cast<PriorBoxParam>(OpManager::GetOpDefParam("PriorBox"));

  const te_caffe::PriorBoxParameter& caffe_param =
      layer_param.prior_box_param();
  // offset
  param.offset = caffe_param.offset();
  // img_size
  if (caffe_param.has_img_h() && caffe_param.has_img_w()) {
    param.img_h = caffe_param.img_h();
    param.img_w = caffe_param.img_w();
  } else if (caffe_param.has_img_size()) {
    param.img_h = caffe_param.img_size();
    param.img_w = caffe_param.img_size();
  } else {
    param.img_h = 0;
    param.img_w = 0;
  }
  // step
  if (caffe_param.has_step_h() && caffe_param.has_step_w()) {
    param.step_h = caffe_param.step_h();
    param.step_w = caffe_param.step_w();
  } else if (caffe_param.has_step()) {
    param.step_h = caffe_param.step();
    param.step_w = caffe_param.step();
  } else {
    param.step_h = 0;
    param.step_w = 0;
  }

  // min_size, max_size
  for (int i = 0; i < caffe_param.min_size_size(); ++i) {
    param.min_size.push_back(caffe_param.min_size(i));
  }
  for (int i = 0; i < caffe_param.max_size_size(); ++i) {
    param.max_size.push_back(caffe_param.max_size(i));
  }

  // variance
  for (int i = 0; i < caffe_param.variance_size(); ++i) {
    param.variance.push_back(caffe_param.variance(i));
  }
  // clip
  param.clip = caffe_param.clip();
  // flip
  param.flip = caffe_param.flip();
  // aspect_ratio
  for (int i = 0; i < caffe_param.aspect_ratio_size(); ++i) {
    param.aspect_ratio.push_back(caffe_param.aspect_ratio(i));
  }

  StaticOp* op = CreateStaticOp(graph, "PriorBox");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}
static bool LoadCaffeResize(StaticGraph* graph, StaticNode* node,
                            const te_caffe::LayerParameter& layer_param) {
  ResizeParam param = any_cast<ResizeParam>(OpManager::GetOpDefParam("Resize"));

  const te_caffe::Resize1_Parameter& caffe_param = layer_param.resize1_param();
  //
  param.scale_h = caffe_param.out_height_scale();
  param.scale_w = caffe_param.out_width_scale();
  param.type = caffe_param.resize_type();
  StaticOp* op = CreateStaticOp(graph, "Resize");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}
static bool LoadCaffeROIPooling(StaticGraph* graph, StaticNode* node,
                                const te_caffe::LayerParameter& layer_param) {
  ROIPoolingParam param =
      any_cast<ROIPoolingParam>(OpManager::GetOpDefParam("ROIPooling"));

  const te_caffe::ROIPoolingParameter& caffe_param =
      layer_param.roi_pooling_param();
  //
  param.pooled_h = caffe_param.pooled_h();
  param.pooled_w = caffe_param.pooled_w();
  param.spatial_scale = caffe_param.spatial_scale();

  StaticOp* op = CreateStaticOp(graph, "ROIPooling");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}
static bool LoadCaffeRPN(StaticGraph* graph, StaticNode* node,
                         const te_caffe::LayerParameter& layer_param) {
  RPNParam param = any_cast<RPNParam>(OpManager::GetOpDefParam("RPN"));

  const te_caffe::RPNParameter& caffe_param = layer_param.rpn_param();
  //
  param.feat_stride = caffe_param.feat_stride();
  param.basesize = caffe_param.basesize();
  param.min_size = caffe_param.boxminsize();
  param.per_nms_topn = caffe_param.per_nms_topn();
  param.post_nms_topn = caffe_param.post_nms_topn();
  param.nms_thresh = caffe_param.nms_thresh();

  for (int i = 0; i < caffe_param.scale_size(); ++i) {
    param.anchor_scales.push_back(caffe_param.scale(i));
  }
  for (int i = 0; i < caffe_param.ratio_size(); ++i) {
    param.ratios.push_back(caffe_param.ratio(i));
  }

  StaticOp* op = CreateStaticOp(graph, "RPN");

  SetOperatorParam(op, param);
  SetOperatorDynamicShape(op);
  SetNodeOp(node, op);

  return true;
}
static bool LoadCaffeDetectionOutput(
    StaticGraph* graph, StaticNode* node,
    const te_caffe::LayerParameter& layer_param) {
  DetectionOutputParam param = any_cast<DetectionOutputParam>(
      OpManager::GetOpDefParam("DetectionOutput"));

  const te_caffe::DetectionOutputParameter& caffe_param =
      layer_param.detection_output_param();

  param.num_classes = caffe_param.num_classes();
  param.confidence_threshold = caffe_param.confidence_threshold();
  param.keep_top_k = caffe_param.keep_top_k();
  param.nms_threshold = caffe_param.nms_param().nms_threshold();
  if (caffe_param.nms_param().has_top_k()) {
    param.nms_top_k = caffe_param.nms_param().top_k();
  }
  StaticOp* op = CreateStaticOp(graph, "DetectionOutput");
  SetOperatorParam(op, param);
  SetOperatorDynamicShape(op);
  SetNodeOp(node, op);

  return true;
}
static bool LoadCaffeReshape(StaticGraph* graph, StaticNode* node,
                             const te_caffe::LayerParameter& layer_param) {
  ReshapeParam param =
      any_cast<ReshapeParam>(OpManager::GetOpDefParam("Reshape"));

  const te_caffe::ReshapeParameter& caffe_param = layer_param.reshape_param();
  // dims
  for (int i = 0; i < caffe_param.shape().dim_size(); ++i) {
    param.dims.push_back(caffe_param.shape().dim(i));
  }

  StaticOp* op = CreateStaticOp(graph, "Reshape");
  SetOperatorParam(op, param);
  SetNodeOp(node, op);

  return true;
}
static EltType ConvertCaffeEltwise(
    te_caffe::EltwiseParameter_EltwiseOp method) {
  if (method == te_caffe::EltwiseParameter_EltwiseOp_PROD)
    return ELT_PROD;
  else if (method == te_caffe::EltwiseParameter_EltwiseOp_MAX)
    return ELT_MAX;

  /* for others, return SUM */

  return ELT_SUM;
}

static bool LoadCaffeEltwise(StaticGraph* graph, StaticNode* node,
                             const te_caffe::LayerParameter& layer_param) {
  const te_caffe::EltwiseParameter& eltwise_param = layer_param.eltwise_param();
  EltwiseParam param =
      any_cast<EltwiseParam>(OpManager::GetOpDefParam("Eltwise"));
  // defalt: SUM
  param.type = ELT_SUM;
  if (eltwise_param.has_operation())
    param.type = ConvertCaffeEltwise(eltwise_param.operation());

  param.caffe_flavor = 1;

  StaticOp* op = CreateStaticOp(graph, "Eltwise");
  SetOperatorParam(op, param);
  SetNodeOp(node, op);

  return true;
}

static bool LoadCaffeDropout(StaticGraph* graph, StaticNode* node,
                             const te_caffe::LayerParameter& layer_param) {
  StaticOp* op = CreateStaticOp(graph, "Dropout");

  SetNodeOp(node, op);

  return true;
}

static bool LoadCaffeAccuracy(StaticGraph* graph, StaticNode* node,
                              const te_caffe::LayerParameter& layer_param) {
  StaticOp* op = CreateStaticOp(graph, "Accuracy");

  SetNodeOp(node, op);

  AddGraphOutputNode(graph, node);

  return true;
}

static bool LoadCaffeConvolution(StaticGraph* graph, StaticNode* node,
                                 const te_caffe::LayerParameter& layer_param) {
  const te_caffe::ConvolutionParameter& conv_param =
      layer_param.convolution_param();

  ConvParam param =
      any_cast<ConvParam>(OpManager::GetOpDefParam("Convolution"));

  if (conv_param.has_kernel_h() && conv_param.has_kernel_w()) {
    param.kernel_h = conv_param.kernel_h();
    param.kernel_w = conv_param.kernel_w();
  } else {
    param.kernel_h = conv_param.kernel_size(0);
    param.kernel_w = conv_param.kernel_size(0);
  }

  if (conv_param.has_stride_h() && conv_param.has_stride_w()) {
    param.stride_h = conv_param.stride_h();
    param.stride_w = conv_param.stride_w();
  } else if (conv_param.stride_size()) {
    param.stride_h = conv_param.stride(0);
    param.stride_w = conv_param.stride(0);
  }

  if (conv_param.has_pad_h() && conv_param.has_pad_w()) {
    param.pad_h = conv_param.pad_h();
    param.pad_w = conv_param.pad_w();
  } else if (conv_param.pad_size()) {
    param.pad_h = conv_param.pad(0);
    param.pad_w = conv_param.pad(0);
  }

  param.output_channel = conv_param.num_output();

  if (conv_param.has_group()) param.group = conv_param.group();

  if (conv_param.dilation_size()) {
    param.dilation_h = conv_param.dilation(0);
    param.dilation_w = conv_param.dilation(0);
  }

  StaticOp* op = CreateStaticOp(graph, "Convolution");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  /* create new Node and tensor for pre-trained weights */

  std::vector<std::string> name_list = {"weight", "bias"};
  std::vector<std::string> layout_list = {"NCHW", "W"};

  LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

  return true;
}

static bool LoadCaffeDeconvolution(
    StaticGraph* graph, StaticNode* node,
    const te_caffe::LayerParameter& layer_param) {
  const te_caffe::ConvolutionParameter& conv_param =
      layer_param.convolution_param();

  DeconvParam param =
      any_cast<DeconvParam>(OpManager::GetOpDefParam("Deconvolution"));

  param.kernel_size = conv_param.kernel_size(0);
  param.stride = conv_param.stride(0);
  param.pad = conv_param.pad(0);
  param.num_output = conv_param.num_output();

  if (conv_param.dilation_size()) {
    param.dilation = conv_param.dilation(0);
  }

  StaticOp* op = CreateStaticOp(graph, "Deconvolution");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  /* create new Node and tensor for pre-trained weights */

  std::vector<std::string> name_list = {"weight", "bias"};
  std::vector<std::string> layout_list = {"NCHW", "C"};

  LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

  return true;
}

static PoolArg ConvertCaffePool(te_caffe::PoolingParameter_PoolMethod method) {
  if (method == te_caffe::PoolingParameter_PoolMethod_AVE)
    return kPoolAvg;
  else if (method == te_caffe::PoolingParameter_PoolMethod_STOCHASTIC)
    return kPoolRand;

  /* for others, return MAX */

  return kPoolMax;
}

static bool LoadCaffePooling(StaticGraph* graph, StaticNode* node,
                             const te_caffe::LayerParameter& layer_param) {
  const te_caffe::PoolingParameter& pool_param = layer_param.pooling_param();

  PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam("Pooling"));

  param.alg = ConvertCaffePool(pool_param.pool());
  if (pool_param.has_kernel_size()) {
    param.kernel_h = pool_param.kernel_size();
    param.kernel_w = pool_param.kernel_size();
  } else if (pool_param.has_kernel_h() && pool_param.has_kernel_w()) {
    param.kernel_h = pool_param.kernel_h();
    param.kernel_w = pool_param.kernel_w();
  }
  param.kernel_shape.resize(2);
  param.kernel_shape[0] = param.kernel_h;
  param.kernel_shape[1] = param.kernel_w;

  param.global = pool_param.global_pooling();

  if (pool_param.has_pad()) {
    param.pad_h = pool_param.pad();
    param.pad_w = pool_param.pad();
  } else if (pool_param.has_pad_h() && pool_param.has_pad_w()) {
    param.pad_h = pool_param.pad_h();
    param.pad_w = pool_param.pad_w();
  }
  param.pads.resize(4);
  param.pads[0] = param.pad_h;
  param.pads[1] = param.pad_w;
  param.pads[2] = param.pad_h;
  param.pads[3] = param.pad_w;

  if (pool_param.has_stride()) {
    param.stride_h = pool_param.stride();
    param.stride_w = pool_param.stride();
  } else if (pool_param.has_stride_h() && pool_param.has_stride_w()) {
    param.stride_h = pool_param.stride_h();
    param.stride_w = pool_param.stride_w();
  }
  param.strides.resize(2);
  param.strides[0] = param.stride_h;
  param.strides[1] = param.stride_w;

  param.caffe_flavor = 1;

  StaticOp* op = CreateStaticOp(graph, "Pooling");

  // SetOperatorDynamicShape(op);
  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}

static bool LoadCaffeInnerProduct(StaticGraph* graph, StaticNode* node,
                                  const te_caffe::LayerParameter& layer_param) {
  const te_caffe::InnerProductParameter& ip_param =
      layer_param.inner_product_param();

  FCParam param = any_cast<FCParam>(OpManager::GetOpDefParam("FullyConnected"));
  param.num_output = ip_param.num_output();

  /* Load weight and bias blob */
  std::vector<std::string> name_list = {"weight", "bias"};
  std::vector<std::string> layout_list = {"HW", "W"};

  LoadCaffeBlob(graph, node, name_list, layout_list, layer_param);

  StaticOp* op = CreateStaticOp(graph, "FullyConnected");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  return true;
}

static bool LoadCaffeBatchNorm(StaticGraph* graph, StaticNode* node,
                               const te_caffe::LayerParameter& layer_param) {
  BatchNormParam param =
      any_cast<BatchNormParam>(OpManager::GetOpDefParam("BatchNormalization"));

  const te_caffe::BatchNormParameter& bn_param = layer_param.batch_norm_param();

  param.eps = bn_param.eps();
  param.caffe_flavor = 1;

  StaticOp* op = CreateStaticOp(graph, "BatchNormalization");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  if (layer_param.blobs_size()) {
    LoadBatchNormBlob(graph, node, layer_param);
  }

  return true;
}

static bool LoadCaffeScale(StaticGraph* graph, StaticNode* node,
                           const te_caffe::LayerParameter& layer_param) {
  ScaleParam param = any_cast<ScaleParam>(OpManager::GetOpDefParam("Scale"));

  const te_caffe::ScaleParameter& scale_param = layer_param.scale_param();

  if (scale_param.has_axis()) param.axis = scale_param.axis();

  if (scale_param.has_num_axes()) param.num_axes = scale_param.num_axes();

  if (scale_param.has_bias_term()) param.bias_term = scale_param.bias_term();

  StaticOp* op = CreateStaticOp(graph, "Scale");

  SetOperatorParam(op, param);

  SetNodeOp(node, op);

  if (layer_param.blobs_size()) {
    LoadScaleBlob(graph, node, layer_param);
  }

  return true;
}

static bool LoadCaffePReLu(StaticGraph* graph, StaticNode* node,
                           const te_caffe::LayerParameter& layer_param) {
  StaticOp* op = CreateStaticOp(graph, "PReLU");

  SetNodeOp(node, op);

  if (layer_param.blobs_size()) {
    LoadPReLuBlob(graph, node, layer_param);
  }

  return true;
}
static bool LoadCaffeLRN(StaticGraph* graph, StaticNode* node,
                         const te_caffe::LayerParameter& layer_param) {
  LRNParam param = any_cast<LRNParam>(OpManager::GetOpDefParam("LRN"));
  const ::te_caffe::LRNParameter& caffe_param = layer_param.lrn_param();

  if (caffe_param.norm_region() ==
      te_caffe::LRNParameter_NormRegion_WITHIN_CHANNEL)
    param.norm_region = LRN_WITHIN_CHANNEL;
  else
    param.norm_region = LRN_ACROSS_CHANNELS;

  param.k = caffe_param.k();
  param.alpha = caffe_param.alpha();
  param.beta = caffe_param.beta();
  param.local_size = caffe_param.local_size();

  StaticOp* op = CreateStaticOp(graph, "LRN");
  SetOperatorParam(op, param);
  SetNodeOp(node, op);

  return true;
}

bool CaffeSerializerRegisterOpLoader(void) {
  SerializerPtr serializer;

  if (!SerializerManager::SafeGet("caffe_single", serializer)) return false;

  CaffeSingle* p_caffe = dynamic_cast<CaffeSingle*>(serializer.get());

  p_caffe->RegisterOpLoadMethod("Data", op_load_t(LoadCaffeInputOp));
  p_caffe->RegisterOpLoadMethod("Input", op_load_t(LoadCaffeInputOp));
  p_caffe->RegisterOpLoadMethod("Convolution", op_load_t(LoadCaffeConvolution));
  p_caffe->RegisterOpLoadMethod("Deconvolution",
                                op_load_t(LoadCaffeDeconvolution));
  p_caffe->RegisterOpLoadMethod("Pooling", op_load_t(LoadCaffePooling));
  p_caffe->RegisterOpLoadMethod("Eltwise", op_load_t(LoadCaffeEltwise));
  p_caffe->RegisterOpLoadMethod("Softmax", op_load_t(LoadCaffeSoftmax));
  p_caffe->RegisterOpLoadMethod("Slice", op_load_t(LoadCaffeSlice));
  p_caffe->RegisterOpLoadMethod("Normalize", op_load_t(LoadCaffeNormalize));
  p_caffe->RegisterOpLoadMethod("SoftmaxWithLoss", op_load_t(LoadCaffeSoftmax));
  p_caffe->RegisterOpLoadMethod("ReLU", op_load_t(LoadCaffeReLu));
  p_caffe->RegisterOpLoadMethod("PReLU", op_load_t(LoadCaffePReLu));
  p_caffe->RegisterOpLoadMethod("InnerProduct",
                                op_load_t(LoadCaffeInnerProduct));
  p_caffe->RegisterOpLoadMethod("Split", op_load_t(LoadCaffeSplit));
  p_caffe->RegisterOpLoadMethod("Concat", op_load_t(LoadCaffeConcat));
  p_caffe->RegisterOpLoadMethod("Dropout", op_load_t(LoadCaffeDropout));
  p_caffe->RegisterOpLoadMethod("Accuracy", op_load_t(LoadCaffeAccuracy));
  p_caffe->RegisterOpLoadMethod("BatchNorm", op_load_t(LoadCaffeBatchNorm));
  p_caffe->RegisterOpLoadMethod("Scale", op_load_t(LoadCaffeScale));
  p_caffe->RegisterOpLoadMethod("LRN", op_load_t(LoadCaffeLRN));
  p_caffe->RegisterOpLoadMethod("Permute", op_load_t(LoadCaffePermute));
  p_caffe->RegisterOpLoadMethod("Flatten", op_load_t(LoadCaffeFlatten));
  p_caffe->RegisterOpLoadMethod("PriorBox", op_load_t(LoadCaffePriorBox));
  p_caffe->RegisterOpLoadMethod("Reshape", op_load_t(LoadCaffeReshape));
  p_caffe->RegisterOpLoadMethod("DetectionOutput",
                                op_load_t(LoadCaffeDetectionOutput));
  p_caffe->RegisterOpLoadMethod("RPN", op_load_t(LoadCaffeRPN));
  p_caffe->RegisterOpLoadMethod("ROIPooling", op_load_t(LoadCaffeROIPooling));
  p_caffe->RegisterOpLoadMethod("Reorg", op_load_t(LoadCaffeReorg));
  p_caffe->RegisterOpLoadMethod("Region", op_load_t(LoadCaffeRegion));
  p_caffe->RegisterOpLoadMethod("Resize", op_load_t(LoadCaffeResize));

  if (!SerializerManager::SafeGet("caffe", serializer)) return false;

  CaffeBuddy* p_buddy = dynamic_cast<CaffeBuddy*>(serializer.get());

  p_buddy->RegisterOpLoadMethod("Data", op_load_t(LoadCaffeInputOp));
  p_buddy->RegisterOpLoadMethod("Input", op_load_t(LoadCaffeInputOp));
  p_buddy->RegisterOpLoadMethod("Convolution", op_load_t(LoadCaffeConvolution));
  p_buddy->RegisterOpLoadMethod("Deconvolution",
                                op_load_t(LoadCaffeDeconvolution));
  p_buddy->RegisterOpLoadMethod("Pooling", op_load_t(LoadCaffePooling));
  p_buddy->RegisterOpLoadMethod("Eltwise", op_load_t(LoadCaffeEltwise));
  p_buddy->RegisterOpLoadMethod("Softmax", op_load_t(LoadCaffeSoftmax));
  p_buddy->RegisterOpLoadMethod("Normalize", op_load_t(LoadCaffeNormalize));
  p_buddy->RegisterOpLoadMethod("Slice", op_load_t(LoadCaffeSlice));
  p_buddy->RegisterOpLoadMethod("SoftmaxWithLoss", op_load_t(LoadCaffeSoftmax));
  p_buddy->RegisterOpLoadMethod("ReLU", op_load_t(LoadCaffeReLu));
  p_buddy->RegisterOpLoadMethod("PReLU", op_load_t(LoadCaffePReLu));
  p_buddy->RegisterOpLoadMethod("InnerProduct",
                                op_load_t(LoadCaffeInnerProduct));
  p_buddy->RegisterOpLoadMethod("Split", op_load_t(LoadCaffeSplit));
  p_buddy->RegisterOpLoadMethod("Concat", op_load_t(LoadCaffeConcat));
  p_buddy->RegisterOpLoadMethod("Dropout", op_load_t(LoadCaffeDropout));
  p_buddy->RegisterOpLoadMethod("Accuracy", op_load_t(LoadCaffeAccuracy));
  p_buddy->RegisterOpLoadMethod("BatchNorm", op_load_t(LoadCaffeBatchNorm));
  p_buddy->RegisterOpLoadMethod("Scale", op_load_t(LoadCaffeScale));
  p_buddy->RegisterOpLoadMethod("LRN", op_load_t(LoadCaffeLRN));
  p_buddy->RegisterOpLoadMethod("Permute", op_load_t(LoadCaffePermute));
  p_buddy->RegisterOpLoadMethod("Flatten", op_load_t(LoadCaffeFlatten));
  p_buddy->RegisterOpLoadMethod("PriorBox", op_load_t(LoadCaffePriorBox));
  p_buddy->RegisterOpLoadMethod("Reshape", op_load_t(LoadCaffeReshape));
  p_buddy->RegisterOpLoadMethod("DetectionOutput",
                                op_load_t(LoadCaffeDetectionOutput));
  p_buddy->RegisterOpLoadMethod("RPN", op_load_t(LoadCaffeRPN));
  p_buddy->RegisterOpLoadMethod("ROIPooling", op_load_t(LoadCaffeROIPooling));
  p_buddy->RegisterOpLoadMethod("Reorg", op_load_t(LoadCaffeReorg));
  p_buddy->RegisterOpLoadMethod("Region", op_load_t(LoadCaffeRegion));
  p_buddy->RegisterOpLoadMethod("Resize", op_load_t(LoadCaffeResize));

  blob_load_map["Convolution"] = LoadConvolutionBlob;
  blob_load_map["Deconvolution"] = LoadDeconvolutionBlob;
  blob_load_map["InnerProduct"] = LoadFullyConnectedBlob;
  blob_load_map["BatchNorm"] = LoadBatchNormBlob;
  blob_load_map["Scale"] = LoadScaleBlob;
  blob_load_map["PReLU"] = LoadPReLuBlob;
  blob_load_map["Normalize"] = LoadNormalizeBlob;

  return true;
}

}  // namespace TEngine
