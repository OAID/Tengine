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
This tool is used for converting models in batches.
Usage:
file struct as belows
Tengine
├── build
    ├──convert_model_batch.py
    ├──onnx_node
    ├──tools/convert_tool/convert_tool
$ python convert_model_batch.py -f "./tools/convert_tool/convert_tool" -m "onnx_node" -s -sp "./convert_result"

Author:
    ycpeng@openailab.com
"""

import argparse
import os
import subprocess

support_onnx_op_list = ["Abs", "Acos", "And", "ArgMax", "ArgMin", "Asin", "Atan", "AveragePool",
                        "Add", "BatchNormalization", "Conv", "ConvTranspose", "Concat", "Clip",
                        "Ceil", "Cos", "Cast", "Dropout", "DepthToSpace", "Div", "Elu", "Exp",
                        "Expand", "Equal", "Flatten", "Floor", "Gemm", "Gather", "Greater", "GlobalAveragePool",
                        "HardSwish", "HardSigmoid", "InstanceNormalization", "Log", "LRN", "Less", "LSTM",
                        "LeakyRelu", "LogSoftmax", "Mul", "Max", "Min", "Mean", "MatMul", "MaxPool",
                        "Neg", "Or", "Pad", "Pow", "PRelu", "Relu", "Resize", "Reshape", "ReduceL2",
                        "ReduceMean", "ReduceLogSumExp", "ReduceLogSum", "ReduceMax", "ReduceMin",
                        "ReduceProd", "ReduceSumSquare", "ReduceSum", "Reciprocal", "Sub",
                        "Selu", "Sqrt", "Slice", "Split", "Shape", "Squeeze", "Scatter", "Sigmoid",
                        "Softmax", "Softplus", "Tanh", "Tile", "Transpose", "Upsample", "Unsqueeze",
                        "Where"]
support_onnx_op_list = [x.lower() for x in support_onnx_op_list]


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")


def parse_args():
    parser = argparse.ArgumentParser(description='convert tools in batch')

    parser.add_argument('-f', help='convert tool path', type=str)
    parser.add_argument('-m', help='model folder path', type=str)
    parser.add_argument('-c', help='convert type', default='onnx', type=str)
    parser.add_argument('-s', help='save convert result', action='store_true')
    parser.add_argument('-sp', help='save result path', default='./convert_results', type=str)
    args = parser.parse_args()
    return args


def convert_model_onnx(convert_tool_path, onnx_model_path):
    """
    convert single model
    :param convert_tool_path:
    :param onnx_model_path:
    :return:
    """
    folder_dir, file_name = os.path.split(onnx_model_path)
    shell_commad = './' if './' not in convert_tool_path else ''
    shell_commad += f"{convert_tool_path} -f onnx -m {onnx_model_path} -o {folder_dir}/onnx.tmfile"
    # print(shell_commad)

    (status, output) = subprocess.getstatusoutput(shell_commad)

    if status != 0:
        if os.path.exists(f"{folder_dir}/onnx.tmfile"):
            shell_commad = f"rm {folder_dir}/onnx.tmfile"
            os.system(shell_commad)
        return False, output
    else:
        return True, output


def main():
    """
    main function
    """
    print("---- batch convert tools ----\n")

    args = parse_args()

    if args.m == None or args.f == None:
        usage_info()
        return None

    print("convert tool path: ", args.f)
    print("model folder path: ", args.m)
    print("convert type     : ", args.c)
    print("save result      : ", args.s)
    print("save folder      : ", args.sp)


    shell_log_dir = args.sp
    if args.s:
        if os.path.exists(shell_log_dir) is not True:
            os.mkdir(shell_log_dir)
        with open(f"{shell_log_dir}/convert_batch_models.txt", 'w') as f:
            f.write(f"{'Model Path':<80} Convert Result\n")

    if args.c.lower() == 'onnx':
        folder_lists = os.listdir(args.m)
        folder_lists.sort()

        for sub_folder in folder_lists:
            sub_folder_path = f"{args.m}/{sub_folder}"
            try:
                op_type = sub_folder.split('_')[1].lower()
            except:
                continue
            if os.path.isdir(sub_folder_path) and op_type in support_onnx_op_list:
                for item in os.listdir(sub_folder_path):
                    if '.onnx' in item:
                        result, log = convert_model_onnx(args.f, f"{sub_folder_path}/{item}")
                        print(f"{sub_folder_path:<80} {result}")
                        if args.s:
                            with open(f"{shell_log_dir}/convert_batch_models.txt", 'a') as f:
                                f.write(f"{sub_folder_path:<80} {result}\n")
                            with open(f"{shell_log_dir}/{sub_folder}.txt", 'w') as f:
                                f.write(log)

if __name__ == '__main__':
    main()
