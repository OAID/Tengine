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
This tool for optimizing the network structure of YOLOv3-tiny from 
https://github.com/ultralytics/yolov3/releases/download/v9.5.0/yolov3-tiny.pt .

0. Export pytorch model to onnx by official models/export.py, e.g.
$ python3 models/export.py --weights yolov3-tiny.pt --img-size 640
1. Update 'Pad' node, e.g.
$ python3 yolov3-tiny-opt.py --input yolov3-tiny.onnx --output yolov3-tiny-opt.onnx

This tool is based on ONNX Framework.

Author:
    qinhj@lsec.cc.ac.cn

Histroy:
2021/05/14  create
"""

import numpy as np
import onnx
import argparse

from onnxsim import simplify


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3-tiny Optimize Tool Parameters')
    
    parser.add_argument('--input', help='input model path', default='./yolov3-tiny.onnx', type=str)  
    parser.add_argument('--output', help='output model path', default='./yolov3-tiny-opt.onnx', type=str)
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    
    args = parser.parse_args()
    return args


def optimize_node_pad(input_node):
    """
    add attribute "pads" to node "Pad" for tengine
    Args:
        input_node: the nodes of ONNX model
    Returns:
        optimized graph nodes(inplace)
    """
    for n in input_node:
        if 'Pad' == n.op_type:
            #print(n)
            ## todo: optimize(search attr "pads" from input nodes)
            n.attribute.extend(onnx.helper.make_attribute(key, value) for key, value in {"pads":(0,0,0,0,0,0,1,1)}.items())
            break
    return input_node


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")
    print("try it again:\n python yolov3-tiny-opt.py -h")


def main():
    """
    main function
    """
    print("---- Tengine YOLOv3-tiny Optimize Tool ----\n")

    args = parse_args()
    if args == None or args.input == None:
        usage_info()
        return None

    print("Input model      : %s" % (args.input))
    print("Output model     : %s" % (args.output))

    # load original onnx model, graph, nodes
    print("[Quant Tools Info]: Step 1, load original onnx model from %s." % (args.input))
    onnx_model = onnx.load(args.input)
    onnx_model, check = simplify(onnx_model,
                                 dynamic_input_shape=args.dynamic,
                                 input_shapes={'images': [1, 3, 640, 640]} if args.dynamic else None)

    graph = onnx_model.graph

    # cut the focus and postprocess nodes
    print("[Quant Tools Info]: Step 2, update 'Pad' node in graph.")
    optimize_node_pad(graph.node)

    # save the new optimize onnx model
    print("[Quant Tools Info]: Step 3, save the new onnx model to %s" % (args.output))
    onnx.save(onnx_model, args.output)

    print("\n---- Tengine YOLOv3-tiny Optimize onnx create success, best wish for your inference has a high accuracy ...\\(^0^)/ ----")


if __name__ == "__main__":
    main()
