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
This tool for optimizing the network structure of YOLOv5s from
https://github.com/ultralytics/yolov5

1. Remove the focus nodes of prepare process;
2. Remove the YOLO detection nodes of postprocess;
3. Fusion the activation HardSwish node replace the Sigmoid and Mul;
4. Update input/output tensor.

This tool is based on ONNX Framework.
Usage:
$ python3 yolov5s-opt.py --input yolov5s.v4.onnx --output yolov5s.v4.opt.onnx --in_tensor 167 --out_tensor 381,420,459
$ python3 yolov5s-opt.py --input yolov5s.v5.onnx --output yolov5s.v5.opt.onnx --in_tensor 167 --out_tensor 397,458,519
$ python3 yolov5s-opt.py --input yolov5s.v5.onnx --output yolov5s-p3p4.opt.onnx --in_tensor 167 --out_tensor 397,458

Author:
    xwwang@openailab.com, initial
    hhchen@openailab.com, update
    qinhj@lsec.cc.ac.cn, update
"""

import numpy as np
import onnx
import argparse

from onnxsim import simplify


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 Optimize Tool Parameters')
    
    parser.add_argument('--input', help='input model path', default='./yolov5s.onnx', type=str)  
    parser.add_argument('--output', help='output model path', default='./yolov5s-opt.onnx', type=str)
    parser.add_argument('--in_tensor', help='input tensor name', default='167', type=str)
    parser.add_argument('--out_tensor', help='output tensor names', default='381,420,459', type=str)
    parser.add_argument('--verbose', action='store_true', help='show verbose info')
    
    args = parser.parse_args()
    return args


args = parse_args()


def cut_focus_output(input_node, in_name, out_name):
    """
    cut the focus and postprocess nodes
    Args:
        input_node: the nodes of ONNX model
        in_name:    input cut tensor value name
        out_name:   output cut tensor value names
    Returns:
        new_nodes:  the new node
    """     
    node_dict = {} # output node name
    for i in range(len(input_node)):
        node_dict[input_node[i].output[0]] = i
    if args.verbose:
        # (key, value): (output node[0] name, index)
        print("[Verbose] node_dict:", node_dict)
    
    # cut output nodes
    output_pass = np.zeros((len(input_node)), dtype=np.int)
    for i in range(len(out_name)):
        output_pass[node_dict[out_name[i]]] = 2
    #if args.verbose:
    #    print("[Verbose] output_pass:", output_pass)

    for i in range(len(input_node)):
        for j in input_node[i].input:
            if j in node_dict:
                if output_pass[node_dict[j]] == 2 or output_pass[node_dict[j]] == 1:
                    output_pass[node_dict[input_node[i].output[0]]] = 1
    if args.verbose:
        print("[Verbose] output_pass:", output_pass)

    # cut focus node
    for i in range(len(output_pass)-1, -1, -1):
        if output_pass[i] == 1:
            del input_node[i]

    # cut input node
    for n in in_name:
        new_nodes = input_node[(node_dict[n] + 1):]

    return new_nodes


def fusion_hardswish(input_node):
    """
    using HardSwish replace the Sigmoid and Mul
    Args:
        input_node: the nodes of ONNX model
    Returns:
        the new node
    """     
    del_list = []
    for i in range(len(input_node) - 1):
        if (input_node[i].op_type == 'Sigmoid' and input_node[i+1].op_type == 'Mul'):
            input_node[i].output[0] = input_node[i+1].output[0]
            input_node[i].op_type = 'HardSwish'
            del_list.append(i + 1)

    for i in range(len(del_list)-1, -1, -1):
        del input_node[del_list[i]]

    return input_node


def keep_or_del_elem(obj, elem_name_list, keep=False):
    """
    keep/delete elem from input objectes
    """
    del_elem_list = []

    for i, n in enumerate(obj):
        if (n.name in elem_name_list and not keep) or (n.name not in elem_name_list and keep):
            del_elem_list.append(i)
    #print("del elem list:", del_elem_list)

    ## delete nodes safely: from end to start
    del_elem_list.reverse()
    [obj.pop(i) for i in del_elem_list]
    return del_elem_list


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")
    print("try it again:\n python yolov5s-opt.py -h")


def main():
    """
    main function
    """
    print("---- Tengine YOLOv5 Optimize Tool ----\n")

    if args == None or args.input == None:
        usage_info()
        return None

    print("Input model      : %s" % (args.input))
    print("Output model     : %s" % (args.output))
    print("Input tensor     : %s" % (args.in_tensor))
    print("Output tensor    : %s" % (args.out_tensor))

    in_tensor = args.in_tensor.split(',')
    out_tensor = args.out_tensor.split(',')

    # load original onnx model, graph, nodes
    print("[Quant Tools Info]: Step 0, load original onnx model from %s." % (args.input))
    onnx_model = onnx.load(args.input)
    onnx_model, check = simplify(onnx_model)

    graph = onnx_model.graph
    print(len(graph.value_info))

    # create the new nodes for optimize onnx model
    old_node  = graph.node
    new_nodes = old_node[:]

    # cut the focus and postprocess nodes
    print("[Quant Tools Info]: Step 1, Remove the focus and postprocess nodes.")
    new_nodes = cut_focus_output(old_node, in_tensor, out_tensor)

    # op fusion, using HardSwish replace the Sigmoid and Mul
    print("[Quant Tools Info]: Step 2, Using hardswish replace the sigmoid and mul.")
    new_nodes = fusion_hardswish(new_nodes)

    # rebuild new model, set the input and outputs nodes
    print("[Quant Tools Info]: Step 3, Rebuild onnx graph nodes.")
    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(new_nodes)

    # get input/output tensor index of value info
    in_tensor_idx = [None] * len(in_tensor)
    out_tensor_idx = [None] * len(out_tensor)
    value = graph.value_info
    for i, v in enumerate(value):
        if v.name in in_tensor:
            in_tensor_idx[in_tensor.index(v.name)] = i
        if v.name in out_tensor:
            out_tensor_idx[out_tensor.index(v.name)] = i
    print("[Quant Tools Info]: Step 4, Update input and output tensor.")

    keep_or_del_elem(onnx_model.graph.input, in_tensor, True)
    for i in in_tensor_idx:
        if i: onnx_model.graph.input.append(value[i])
    keep_or_del_elem(onnx_model.graph.output, out_tensor, True)
    for i in out_tensor_idx:
        if i: onnx_model.graph.output.append(value[i])

    # save the new optimize onnx model
    print("[Quant Tools Info]: Step 5, save the new onnx model to %s." % (args.output))
    onnx.save(onnx_model, args.output)

    print("\n---- Tengine YOLOv5s Optimize onnx create success, best wish for your inference has a high accuracy ...\\(^0^)/ ----")


if __name__ == "__main__":
    main()
