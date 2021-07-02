# -*- coding: utf-8 -*-

# OPEN AI LAB is pleased to support the open source community by supporting Tengine available.
#
# Copyright (C) 2021 OPEN AI LAB. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


"""
This tool is mainly for optimizing the network structure of nanodet_m.ckpt from 
https://github.com/RangiLyu/nanodet .

Preparation:
1. Export pytorch model to onnx by official tools/export_onnx.py, e.g.
$ python tools/export_onnx.py --cfg_path config/nanodet-m.yml --model_path nanodet_m.ckpt

Optimization:
1. Load onnx model and just simplify it:
$ python3 nanodet_m-opt.py --input nanodet_m.onnx --output nanodet_m-opt.onnx

Optimization(not recommended):
1. Updata the output shape in all distance prediction branches from [1, *, 32] to [1, *, 4, 8];
2. Add additional "Softmax" node in the end of all distance prediction branches with axis=-1;
3. Update output tensor name(from "dis_pred_stride_*" to "dis_sm_stride_*", in which "sm" is
short for "softmax") and shape(from [1, *, 32] to [1, *, 4, 8] for later integral);
4. Check and simplify new onnx model.
$ python3 nanodet_m-opt.py --input nanodet_m.onnx --output nanodet_m-opt.onnx --softmax --const 893,915,937

This tool is based on ONNX Framework.

Author:
    qinhj@lsec.cc.ac.cn

Histroy:
2021/06/26  create
"""

import argparse
import onnx

from onnxsim import simplify


def parse_args():
    parser = argparse.ArgumentParser(description='NanoDet-m Optimize Tool Parameters')
    
    parser.add_argument('--input', type=str, default='nanodet_m.onnx', help='input model path')  
    parser.add_argument('--output', type=str, default='nanodet_m-opt.onnx', help='output model path')
    parser.add_argument('--const', type=str, default='893,915,937', help='constant(nodes) for final reshape node in distance prediction branch')
    parser.add_argument("--softmax", action='store_true', default=False, help="add additional softmax node to distance prediction branch")
    
    args = parser.parse_args()
    return args


def optimize_node_shape(nodes, names):
    """
    optimize input constant nodes of final reshape nodes in distance prediction branch
    Args:
        nodes: the graph.node of ONNX model
        names: target constant node name list
    Returns:
        optimized graph nodes(inplace)
    """
    ## new shape value for "Constant" nodes
    t = onnx.helper.make_tensor('', onnx.TensorProto.INT64, [4], [1, 4, 8, -1])
    t = [onnx.helper.make_attribute(key, value) for key, value in {"value": t}.items()]
    ## new attribute for "Transpose" nodes
    a = [onnx.helper.make_attribute(key, value) for key, value in {"perm":(0,3,1,2)}.items()]

    reshape_output = []
    for i, n in enumerate(nodes):
        if 'Constant' == n.op_type and n.output[0] in names:
            ## replace attr with new one
            n.attribute.pop()
            n.attribute.extend(t)
            #print(n)
            continue
        if 'Reshape' == n.op_type and set(names).intersection(n.input):
            ## cache output tensor name of reshape node
            reshape_output.extend(n.output)
            #print(n)
            continue
        if 'Transpose' == n.op_type and n.input[0] in reshape_output:
            ## replace attr with new one
            n.attribute.pop()
            n.attribute.extend(a)
            #print(n)
            continue
    return nodes


def optimize_output_tensor(output):
    """
    optimize output tensor name and shape
    Args:
        output: the graph.output of ONNX model
    Returns:
        optimized graph output(inplace)
    """
    for o in output:
        if "dis_pred_stride_" in o.name:
            _d = o.type.tensor_type.shape.dim
            ## kick out the last dim: 32
            d2 = _d.pop(2)
            ## add two new dims: 4, 8
            d2.dim_value = 4
            _d.append(d2)
            d2.dim_value = 8
            _d.append(d2)
            ## update output name
            o.name = o.name.replace("dis_pred_stride_", "dis_sm_stride_")
    return output


def optimize_add_softmax(nodes):
    """
    add additional softmax node in the end of all distance prediction branches
    Args:
        nodes: the graph.node of ONNX model
    Returns:
        optimized graph nodes(inplace)
    """
    for n in nodes:
        if 'Transpose' == n.op_type and "dis_pred_stride_" in n.output[0]:
            ## add additional softmax node
            _input = n.output[0]
            _output = _input.replace("dis_pred_stride_", "dis_sm_stride_")
            n_sm = onnx.helper.make_node('Softmax', inputs=[_input], outputs=[_output], axis=-1)
            nodes.append(n_sm)
    return nodes


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")
    print("try it again:\n python nanodet_m-opt.py -h")


def main():
    """
    main function
    """
    print("---- Tengine NanoDet-m Optimize Tool ----\n")

    args = parse_args()
    if args == None or args.input == None:
        usage_info()
        return None

    print(" Input model path    : %s" % (args.input))
    print("Output model path    : %s" % (args.output))

    # load original onnx model, graph, nodes
    print("[Opt Tools Info]: Step 0, load original onnx model from %s." % (args.input))
    onnx_model = onnx.load(args.input)

    if args.softmax:
        constant_shape_list = args.const.split(',')

        # update input constant nodes
        print("[Opt Tools Info]: Step 1, update the output shape in all dis_pred branches.")
        optimize_node_shape(onnx_model.graph.node, constant_shape_list)

        # add additional softmax nodes
        print("[Opt Tools Info]: Step 2, add Softmax node in the end of all dis_pred branche.")
        optimize_add_softmax(onnx_model.graph.node)

        # update output tensor name and shape
        print("[Opt Tools Info]: Step 3, update output tensor name and shape.")
        optimize_output_tensor(onnx_model.graph.output)

    # do check and simplify the onnx model
    print("[Opt Tools Info]: Step 4, check and simplify the new onnx model.")
    onnx_model, check = simplify(onnx_model)

    # save the new optimize onnx model
    print("[Opt Tools Info]: Step 5, save the new onnx model to %s" % (args.output))
    onnx.save(onnx_model, args.output)

    print("\n---- Tengine NanoDet-m Optimize onnx create success, best wish for your inference has a high accuracy ...\\(^0^)/ ----")


if __name__ == "__main__":
    main()
