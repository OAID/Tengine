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
https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.onnx.

1. Remove the focus nodes of prepare process;
2. Remove the YOLO detection nodes of postprocess;
3. Fusion the activation HardSwish node replace the Sigmoid and Mul.

This tool is based on ONNX Framework.

Author:
    xwwang@openailab.com, initial
    hhchen@openailab.com, update
"""

import numpy as np
import onnx
import argparse

from onnxsim import simplify


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 Optimize Tool Parameters')
    
    parser.add_argument('--input', help='input model path', default='./yolov5s.onnx', type=str)  
    parser.add_argument('--output', help='output model path', default='./yolov5s-opt.onnx', type=str)
    parser.add_argument('--in_cut_name', help='input cut node name', default='167', type=str)
    parser.add_argument('--output_num', help='output num', default=3, type=int)
    parser.add_argument('--out_cut_names', help='output cut node names', default='381,420,459', type=str)
    
    args = parser.parse_args()
    return args


args = parse_args()


def cut_focus_output(input_node, in_name, out_name, out_num):
    """
    cut the focus and postprocess nodes
    Args:
        input_node: input_node: the nodes of ONNX model
        in_name:    input cut node name
        out_name:   output cut node names
        out_num:    output num
    Returns:
        new_nodes:  the new node
    """     
    node_dict = {}
    for i in range(len(input_node)):
        node_dict[input_node[i].output[0]] = i
    
    # cut output nodes
    output_pass = np.zeros((len(input_node)), dtype=np.int)
    for i in range(out_num):
        output_pass[node_dict[out_name[i]]] = 2

    for i in range(len(input_node)):
        for j in input_node[i].input:
            if j in node_dict:
                if output_pass[node_dict[j]] == 2 or output_pass[node_dict[j]] == 1:
                    output_pass[node_dict[input_node[i].output[0]]] = 1

    # cut focus node
    for i in range(len(output_pass)-1, -1, -1):
        if output_pass[i] == 1:
            del input_node[i]
            
    new_nodes = input_node[(node_dict[in_name] + 1):]

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
        if(input_node[i].op_type == 'Sigmoid' and input_node[i+1].op_type == 'Mul'):
            input_node[i].output[0] = input_node[i+1].output[0]
            input_node[i].op_type = 'HardSwish'
            del_list.append(i + 1)

    for i in range(len(del_list)-1, -1, -1):
        del input_node[del_list[i]]

    return input_node


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
    print("Input node       : %s" % (args.in_cut_name))
    print("Output nodes     : %s" % (args.out_cut_names))
    print("Output node num  : %s\n" % (args.output_num))

    in_cut_name = args.in_cut_name
    output_num = args.output_num
    out_cut_names = args.out_cut_names.split(',')

    # load original onnx model, graph, nodes
    print("[Quant Tools Info]: Step 0, load original onnx model from %s." % (args.input))
    onnx_model = onnx.load(args.input)
    onnx_model, check = simplify(onnx_model)

    graph = onnx_model.graph
    old_node  = graph.node
        
    # create the new nodes for optimize onnx model
    new_nodes = old_node[:]

    # cut the focus and postprocess nodes
    print("[Quant Tools Info]: Step 1, Remove the focus and postprocess nodes.")
    new_nodes = cut_focus_output(old_node, in_cut_name, out_cut_names, output_num)

    # op fusion, using HardSwish replace the Sigmoid and Mul
    print("[Quant Tools Info]: Step 2, Using hardswish replace the sigmoid and mul.")
    new_nodes = fusion_hardswish(new_nodes)

    # rebuild new model, set the input and ouptus nodes
    print("[Quant Tools Info]: Step 3, Rebuild new onnx model.")
    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(new_nodes)

    conv0_node = onnx_model.graph.node[0]
    conv0_node.input[0] = 'images'
    graph = onnx_model.graph

    out_back = onnx_model.graph.output[0]

    del onnx_model.graph.output[:]

    onnx_model.graph.output.append(out_back)
    onnx_model.graph.output.append(out_back)
    onnx_model.graph.output.append(out_back)

    for i in range(output_num):
        onnx_model.graph.output[i].name = out_cut_names[i]

    # save the new optimize onnx model
    print("[Quant Tools Info]: Step 4, save the new onnx model to %s" % (args.output))
    onnx.save(onnx_model, args.output)

    print("\n---- Tengine YOLOv5s Optimize onnx create success, best wish for your inference has a high accuracy ...\\(^0^)/ ----")


if __name__ == "__main__":
    main()
