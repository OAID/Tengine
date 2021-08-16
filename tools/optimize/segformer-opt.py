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
This tool for optimizing the network structure of Segformer from
https://github.com/NVlabs/SegFormer
and this tool reference from 
https://github.com/OAID/Tengine/blob/tengine-lite/tools/optimize/yolov5s-opt.py

1. Fusion the Gelu node replace the Div-Erf-Add-Mul-Mul;
2. Fusion the LayerNorm(affine=True) node replace the ReduceMean-Sub-Pow-ReduceMean-Add-Sqrt-Div-Mul-Add;
3. Update input/output tensor.

This tool is based on ONNX Framework.
Usage:
$ python3 segformer-opt.py --input segformer.b0.512x1024.city.onnx --output segformer.b0.512x1024.city.opt.onnx --in_tensor x --out_tensor 1729


"""

import numpy as np
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import argparse
import struct
from onnxsim import simplify


def parse_args():
    parser = argparse.ArgumentParser(description='Segformer Optimize Tool Parameters')
    
    parser.add_argument('--input', help='input model path', default='./segformer.b0.512x1024.city.onnx', type=str)  
    parser.add_argument('--output', help='output model path', default='./segformer.b0.512x1024.city.opt.onnx', type=str)
    parser.add_argument('--in_tensor', help='input tensor name', default='x', type=str)
    parser.add_argument('--out_tensor', help='output tensor names', default='1729', type=str)
    parser.add_argument('--verbose', action='store_true', help='show verbose info')
    
    args = parser.parse_args()
    return args

args = parse_args()

def fusion_gelu(input_node):  
    del_list = []
    gelu_count = 1
    for i in range(len(input_node)-1):
        #fuse Div-Erf-Add-Mul-Mul=====>Gelu
        if (input_node[i].op_type == 'Div' and input_node[i+1].op_type == 'Erf'
            and input_node[i+2].op_type == 'Add' and input_node[i+3].op_type == 'Mul'
            and input_node[i+4].op_type == 'Mul'):
            input_node[i].output[0] = input_node[i+4].output[0]
            input_node[i].op_type = 'Gelu'
            input_node[i].name = 'Gelu_'+str(gelu_count) #change node name
            gelu_count += 1
            
            del input_node[i].input[1]
            del_list.append(i + 1)
            del_list.append(i + 2)
            del_list.append(i + 3)
            del_list.append(i + 4)
    for i in range(len(del_list)-1, -1, -1):
        del input_node[del_list[i]]

    return input_node

#layarnorm's affine of segformer is True
def fusion_layernorm(input_node,init_dict):  
    del_list = []
    layernorm_count = 1
    for i in range(len(input_node)-1):
        #fuse ReduceMean-Sub-Pow-ReduceMean-Add-Sqrt-Div-Mul-Add=====>layernorm
        if (input_node[i].op_type == 'ReduceMean' and input_node[i+1].op_type == 'Sub'
            and input_node[i+2].op_type == 'Cast' and input_node[i+3].op_type == 'Pow'
            and input_node[i+4].op_type == 'ReduceMean' and input_node[i+5].op_type == 'Add'
            and input_node[i+6].op_type == 'Sqrt' and input_node[i+7].op_type == 'Div'
            and input_node[i+8].op_type == 'Mul' and input_node[i+9].op_type == 'Add'):

            new_node = onnx.helper.make_node(
                'LayerNorm',
                name = 'LayerNorm_' + str(layernorm_count),
                inputs = [input_node[i].input[0], input_node[i+8].input[1],input_node[i+9].input[1]],
                outputs = [input_node[i+9].output[0]],
                epsilon = struct.unpack('f',init_dict[input_node[i+5].input[1]].raw_data)[0],
                affine = 1
            )
            layernorm_count += 1
            input_node.remove(input_node[i])
            input_node.insert(i, new_node)

            del_list.append(i + 1)
            del_list.append(i + 2)
            del_list.append(i + 3)
            del_list.append(i + 4)
            del_list.append(i + 5)
            del_list.append(i + 6)
            del_list.append(i + 7)
            del_list.append(i + 8)
            del_list.append(i + 9)
        #TODO:add affine=False
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
    print("try it again:\n python segformer-opt.py -h")


def main():
    """
    main function
    """
    print("---- Tengine Segformer Optimize Tool ----\n")

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
    init_dict = dict()
    for init in onnx_model.graph.initializer:
        init_dict[init.name]=init
    
    # create the new nodes for optimize onnx model
    old_node  = graph.node
    new_nodes = old_node[:-8]

    # op fusion
    print("[Quant Tools Info]: Step 1, fuse Gelu and LayerNorm.")
    new_nodes = fusion_gelu(new_nodes)
    new_nodes = fusion_layernorm(new_nodes,init_dict)

    # rebuild new model, set the input and outputs nodes
    print("[Quant Tools Info]: Step 2, Rebuild onnx graph nodes.")
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
    print("[Quant Tools Info]: Step 3, Update input and output tensor.")

    keep_or_del_elem(onnx_model.graph.input, in_tensor, True)
    for i in in_tensor_idx:
        if i: onnx_model.graph.input.append(value[i])
    keep_or_del_elem(onnx_model.graph.output, out_tensor, True)
    for i in out_tensor_idx:
        if i: onnx_model.graph.output.append(value[i])

    # save the new optimize onnx model
    print("[Quant Tools Info]: Step 4, save the new onnx model to %s." % (args.output))
    onnx.save(onnx_model, args.output)

    print("\n---- Tengine Segformer Optimize onnx create success, best wish for your inference has a high accuracy ...\\(^0^)/ ----")


if __name__ == "__main__":
    main()
