#include <time.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "logger.hpp"
#include "node_ops.hpp"

#include "graph.hpp"
#include "tensor_mem.hpp"

#include "acl_driver.hpp"
#include "acl_lrn.hpp"
#include "operator/lrn.hpp"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"

#include "CL/cl2.hpp"
#include "utils/Utils.h"

using namespace arm_compute;

using namespace utils;

namespace TEngine {

struct ACLLrnArg {
  CLTensor input;
  CLTensor output;
  NormType norm_type;

  CLNormalizationLayer cllrn;
};

bool ACLLrnOps::Prerun(Node *node) {
  // CLScheduler::get().default_init();
  CLScheduler::get().default_init();

  LRN *lrn_op = dynamic_cast<LRN *>(node->GetOp());

  LRNParam *param = lrn_op->GetParam();
  ACLLrnArg *arg = new ACLLrnArg();
  /* input */
  Tensor *input_tensor = node->GetInputTensor(0);
  float alpha = param->alpha;
  float beta = param->beta;
  float bias = param->k;
  int local_size = param->local_size;
  int IN_MAP = param->norm_region;

  if (0 != IN_MAP)  // now only support LRN_ACROSS_CHANNELS;
    return false;

  arg->norm_type = (NormType)0;

  NormalizationLayerInfo norm_info(arg->norm_type, local_size, alpha, beta,
                                   bias);

  TShape &ishape = input_tensor->GetShape();

  unsigned int input_w = ishape.GetW();
  unsigned int input_h = ishape.GetH();
  unsigned int input_c = ishape.GetC();

  TensorShape _ishape(input_w, input_h, input_c);

  arg->input.allocator()->init(TensorInfo(_ishape, 1, DataType::F32));

  /* output */
  Tensor *output_tensor = node->GetOutputTensor(0);
  TShape &oshape = output_tensor->GetShape();

  int mem_size = output_tensor->GetTotalSize();
  void *addr = std::malloc(mem_size);
  set_tensor_mem(output_tensor, addr, mem_size, std::free);

  unsigned int out_w = oshape.GetW();
  unsigned int out_h = oshape.GetH();
  unsigned int out_c = oshape.GetC();
  TensorShape _oshape(out_w, out_h, out_c);
  arg->output.allocator()->init(TensorInfo(_oshape, 1, DataType::F32));

  // allocate the buffer
  arg->input.allocator()->allocate();
  arg->output.allocator()->allocate();

  arg->cllrn.configure(&arg->input, &arg->output, norm_info);

  (*node)["ACLLrnArg"] = arg;

  return true;
}

bool ACLLrnOps::Run(Node *node) {
  /* input */
  Tensor *itensor = node->GetInputTensor(0);
  TShape &ishape = itensor->GetShape();
  float *input_org = (float *)get_tensor_mem(itensor);
  int isize = ishape.GetSize();

  ACLLrnArg *arg = any_cast<ACLLrnArg *>(node->GetAttr("ACLLrnArg"));
  arg->input.map();
  float *ibuf = reinterpret_cast<float *>(arg->input.buffer());
  memcpy(ibuf, input_org, isize * 4);
  arg->input.unmap();

  /* output */
  Tensor *output_tensor = node->GetOutputTensor(0);
  TShape &output_shape = output_tensor->GetShape();
  int out_size = output_shape.GetSize();
  float *output_buf = (float *)get_tensor_mem(output_tensor);

  arg->cllrn.run();

  const cl::Buffer &cl_buf = arg->output.cl_buffer();

  cl::copy<float *>(cl_buf, output_buf, output_buf + out_size);

  return true;
}

bool ACLLrnOps::Postrun(Node *node) {
  ACLLrnArg *arg = any_cast<ACLLrnArg *>(node->GetAttr("ACLLrnArg"));

  arg->input.allocator()->free();
  arg->output.allocator()->free();

  delete arg;

  return true;
}

}  // namespace TEngine
