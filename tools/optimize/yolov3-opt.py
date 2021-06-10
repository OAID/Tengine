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
This tool for optimizing the network structure of YOLOv3 from 
https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3.pt .

1. Load original model as simplify format;
2. Cut target nodes;
3. Update input and output nodes;

Usage:
$ python3 yolov3-opt.py --input yolov3.onnx --output yolov3-opt.onnx --cut "Conv_156"

This tool is based on ONNX Framework.

Author:
    qinhj@lsec.cc.ac.cn

Histroy:
2021/05/19  create
"""

import numpy as np
import onnx
import argparse

from onnxsim import simplify


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 Optimize Tool Parameters')
    
    parser.add_argument('--input', help='input model path', default='./yolov3.onnx', type=str)  
    parser.add_argument('--output', help='output model path', default='./yolov3-opt.onnx', type=str)
    parser.add_argument('--cut', help='cut node names', default='Conv_156', type=str)
    
    args = parser.parse_args()
    return args


"""
Note:
1) both node.remove and node.pop are not safe during iter;
"""
def cut_nodes(input_node, cut_names):
    """
    del nodes from input nodes
    Args:
        input_node: the nodes of ONNX model
    Returns:
        optimized graph nodes(inplace) and delet node names
    """
    del_node_name = set(cut_names)
    del_node_list = []

    for i, n in enumerate(input_node):
        if n.name in del_node_name or set(n.input).intersection(del_node_name):
            del_node_list.append(i)
            del_node_name.add(n.name)
            [del_node_name.add(o) for o in n.output]
    print("del node name:", del_node_name)
    print("del node list:", del_node_list)

    ## delete nodes safely: from end to start
    del_node_list.reverse()
    [input_node.pop(i) for i in del_node_list]
    return del_node_name


def cut_nodes_input_output(node, cut_names):
    """
    del nodes from input nodes
    Args:
        node: the nodes of ONNX model
    Returns:
        optimized graph nodes(inplace)
    """
    for i in range(len(node) - 1, -1, -1):
        if node[i].name in cut_names:
            node.pop(i)


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")
    print("try it again:\n python yolov3-opt.py -h")


def main():
    """
    main function
    """
    print("---- Tengine YOLOv3 Optimize Tool ----\n")

    args = parse_args()
    if args == None or args.input == None:
        usage_info()
        return None

    print("Input model      : %s" % (args.input))
    print("Output model     : %s" % (args.output))
    print("Cut node names   : %s" % (args.cut))

    # load original onnx model, graph, nodes
    print("[Quant Tools Info]: Step 1, load original onnx model from %s." % (args.input))
    onnx_model = onnx.load(args.input)
    onnx_model, check = simplify(onnx_model)

    graph = onnx_model.graph
    #print(graph.input)
    #print(graph.output)

    # cut the target nodes
    print("[Quant Tools Info]: Step 2, cut nodes '%s' node in graph." % (args.cut))
    cut_node_name = args.cut.split(',')
    del_node_name = cut_nodes(graph.node, cut_node_name)

    # update graph input and output nodes
    print("[Quant Tools Info]: Step 3, update the input and output nodes")
    cut_nodes_input_output(graph.input, del_node_name)
    cut_nodes_input_output(graph.output, del_node_name)

    # save the new optimize onnx model
    print("[Quant Tools Info]: Step 4, save the new onnx model to %s" % (args.output))
    onnx.save(onnx_model, args.output)

    print("\n---- Tengine YOLOv3 Optimize onnx create success, best wish for your inference has a high accuracy ...\\(^0^)/ ----")


if __name__ == "__main__":
    main()
