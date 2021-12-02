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
This tool is used for aligning the outputs of tengine and onnx.
1. pip install -r requirements.txt
2. install pytengine
3. prepare onnx model file and tengine model file


Example:
$ python align_with_onnx.py --m mnist.onnx  --tm mnist.tmfile --a
$ python align_with_onnx.py --m mnist.onnx  --tm mnist.tmfile --a --s

This tool is based on ONNX Framework.

Author:
    ycpeng@openailab.com

Histroy:
2021/08/09  create

"""
import onnx
import onnxruntime
import argparse
import numpy as np
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(description='align tools: align the output of onnx and tengine by layers')
    parser.add_argument('--m', help='input onnx model path', type=str)
    parser.add_argument('--s', action='store_true', help='save onnx output text')
    parser.add_argument('--sp', default="./output_onnx", help='output onnx model path', type=str)
    parser.add_argument('--a', action='store_true', help='align outputs by layer')
    parser.add_argument('--to', help='path of tengine output text', type=str)
    parser.add_argument('--tm', help='tengine model', type=str)
    args = parser.parse_args()
    return args


def check_make_folder(folder_path):
    if os.path.exists(folder_path) == False:
        os.makedirs(folder_path)


class AlignOnnx():
    def __init__(self, onnx_path, is_export_onnx_output,
                 export_path, align_by_layer=False,
                 tengine_output_path=None,
                 tengine_model=None):
        """
        initial function
        :param onnx_path: onnx model path
        :param is_export_onnx_output: if export onnx output to folder
        :param export_path: the path of onnx output
        :param align_by_layer: align onnx and tengine  by layer
        :param tengine_output_path: the path of tengine output
        :param tengine_model: tengine model path
        """
        self.is_export_onnx_output = is_export_onnx_output
        self.export_path = export_path
        self.align_by_layer = align_by_layer
        self.tengine_output_path = tengine_output_path
        self.tengine_model = tengine_model

        self.onnx_model = onnx.load(onnx_path)

    def __call__(self):
        """
        infer model, save results, compare results
        :return:
        """
        # set the onnx outputs, for getting the output of all layers
        const_name = []
        for item in self.onnx_model.graph.initializer:
            const_name.append(item.name)
        for one_node in self.onnx_model.graph.node:
            for output in one_node.output:
                self.onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            for input in one_node.input:
                if input not in const_name:
                    self.onnx_model.graph.output.extend([onnx.ValueInfoProto(name=input)])
        session = onnxruntime.InferenceSession(self.onnx_model.SerializeToString())

        # set inputs , you can change your input here
        input_shape = session.get_inputs()[0].shape
        print(input_shape)
        if self.tengine_model is not None:
            input_array = np.random.randn(*input_shape).astype(np.float32)
        else:
            input_array = np.ones(input_shape).astype(np.float32)

        # get onnx outputs
        outputs, result = self._onnx_inference(session, input_array)
        print("onnx inference over")
        # save onnx outputs
        if self.is_export_onnx_output:
            self._save_result(outputs, result, const_name)
            print("onnx export ok")
        print("-------------------------------------------------")

        # align with tengine file: the compare results will save to export_path
        if self.align_by_layer:
            # get tengine output if enabled
            if self.tengine_model is not None:
                if self._tengine_inference(input_array, input_shape) == False:
                    print("tengine model inference failed!")
                    return
                print('tengine inference over')
                self.tengine_output_path = './output'

            self._align_by_layer(outputs, result)

    def _save_result(self, outputs, result, const_name):
        """
        save results of onnx
        :param outputs: list of outputs name
        :param result: onnx inference result
        :param const_name: onnx const name that should not contain in output node
        :return:
        """
        check_make_folder(self.export_path)
        print("-------------------------------------------------")
        print("export onnx output to text , please wait a moment")
        for one_node in tqdm(self.onnx_model.graph.node):
            sufix = self.export_path + '/' + one_node.name.replace('/','-')
            for i, input in enumerate(one_node.input):
                if input not in const_name:
                    self._write_data(sufix + f'_in_blob_data.txt', result[outputs.index(input)])
                    # self._write_data(sufix+f'_in[{i}]_blob_data.text',result[outputs.index(input)])
            for output in one_node.output:
                self._write_data(sufix + f'_out_blob_data.txt', result[outputs.index(output)])

    def _write_data(self, text_path, data):
        """
        too difficult to write, I don't want to do this again.  :(
        If anyone has better way to realizing, pls tell me or PR.
        """
        shape_str = '{' + ' '.join([str(i) for i in data.shape]) + '}'
        mat_shape = data.shape
        data = data.flatten()
        data = np.round(data, decimals=4)

        if np.float32 == data.dtype:
            data_type = 'fp32'

        write_string = f"Shape is " + shape_str + f", data type is {data_type}\n"
        for i in range(mat_shape[0]):

            # lack dim1 and dim5
            dim = len(mat_shape)
            if dim != 2 and dim != 3 and dim != 4:
                continue
            write_data_batch = data[i * np.prod(mat_shape[1:]):i + 1 * np.prod(mat_shape[1:])]

            for c in range(mat_shape[1]):
                if dim == 2:
                    write_string += f"{write_data_batch[c]: .4f} "
                if dim == 3:
                    h = mat_shape[2]
                    write_string += f"\tChannel {c}:\n"
                    write_string += f"\t\t{' '.join([f'{i: .4f}' for i in write_data_batch[c * h:(c + 1) * h]])}\n"
                if dim == 4:
                    h, w = mat_shape[2], mat_shape[3]
                    write_data_channel = write_data_batch[c * np.prod(mat_shape[2:]):(c + 1) * np.prod(mat_shape[2:])]
                    write_string += f"Batch {i}:\n"
                    write_string += f"\tChannel {c}:\n"
                    for hi in range(h):
                        write_string += f"\t\t{' '.join([f'{i: .4f}' for i in write_data_channel[hi * w:(hi + 1) * w]])}\n"

        with open(text_path, 'w') as f:
            f.writelines(write_string)

    def _onnx_inference(self, session, input_array):
        """
        onnx inference as the function name
        :param session:
        :param input_array:
        :return:
        """
        # inference onnxmodel
        input_name = session.get_inputs()[0].name
        output_name = list(set([i.name for i in session.get_outputs()]))
        result = session.run(output_name, {input_name: input_array})
        return output_name, result


    def _tengine_inference(self, input_array, input_shape):
        """
        tengine inference
        :param input_array:
        :param input_shape:
        :return:
        """
        os.environ['TG_DEBUG_DATA'] = '1'
        from tengine import tg
        try:
            graph = tg.Graph(None, 'tengine', self.tengine_model)
            input_tensor = graph.getInputTensor(0, 0)
            input_tensor.shape = input_shape
            input_image = input_tensor.ascontiguousarray(input_array)
            input_tensor.buf = input_image
            input_tensor.setData(input_image)
            graph.preRun()
            graph.run(1)
            graph.postRun()
            del os.environ['TG_DEBUG_DATA']
            return True
        except:
            print("run tengine model faield, please check your model or input")
            del os.environ['TG_DEBUG_DATA']
            return False


    def _calc_distance(self, array_1, array_2, calc_type='L1'):
        """
        calculate the distance
        :array_1: input array1
        :array_2: input array2
        :calc_type: select L1 or L2
        return : the similarity between two input arrays
        """
        sim = 0
        if calc_type == 'L1':
            sim = np.abs(np.sum(np.abs(array_1) - np.abs(array_2)))
        if calc_type == 'L2':
            array_1 = array_1.astype(np.float64)
            array_2 = array_2.astype(np.float64)
            num = np.dot(array_1.flatten(), array_2.flatten())
            denom = np.linalg.norm(array_1.flatten(), ord=2) * np.linalg.norm(array_2.flatten(), ord=2)

            sim = (num / denom) if denom != 0 else 0
        return sim

    def _align_by_layer(self, outputs, result):
        """
        align output by layers
        :param outputs:
        :param result:
        :return:
        """
        print('-------------------------------------------------')
        tm_txt_list = os.listdir(self.tengine_output_path)
        check_make_folder(self.export_path)

        with open(f"{self.export_path}/compare.txt", "w") as f:
            out_info = f"{'TEXT':<50}{'L1 DISTANCE':<20}{'L2 DISTANCE':<20}"
            f.write(f"{out_info}\n")
            print(out_info)

            for one_node in self.onnx_model.graph.node:
                text_file_name = [(one_node.name + "_out_blob_data.txt").replace('/','-'), (one_node.name + "_in_blob_data.txt").replace('/','-')]

                def compare_and_write(file_name, compare_input=False):
                    if file_name in tm_txt_list:
                        # read tengine layer output
                        tm_ndarray = self._txt_2_ndarray(f"{self.tengine_output_path}/{file_name}")
                        if tm_ndarray is None:
                            print(f"load {self.tengine_output_path}/{file_name} failed")
                            return
                        # get correspond onnx layer output
                        name = one_node.input if compare_input else one_node.output
                        try:
                            sf_ndarray = result[outputs.index(name[0])]
                        except:
                            print(f"load onnx output failed {name}")
                            return

                        # calculate distance
                        L1_distance = self._calc_distance(tm_ndarray, sf_ndarray, 'L1')
                        L2_distance = self._calc_distance(tm_ndarray, sf_ndarray, 'L2')

                        # write to text
                        out_info = f"{file_name:<50}{L1_distance:<20.5}{L2_distance:0<10.6f}"
                        f.write(f"{out_info}\n")
                        print(out_info)
                # compare input
                compare_and_write(text_file_name[1], True)
                # compare output
                compare_and_write(text_file_name[0])

    def _txt_2_ndarray(self, txt_path):
        """
        :read the text that contains layer output
        return : the ndarray of the text data
        """
        with open(txt_path, 'r') as f:
            row1 = f.readline()
            index_1 = row1.index("{") + 1
            index_2 = row1.index("}")
            array_shape = row1[index_1:index_2].split(' ')
            content = f.read()
            content = content.split()
            data = [float(x) for x in content if ':' not in x and 'a' not in x and 'Dim' not in x]
        try:
            np_array = np.array(data, dtype=np.float64).reshape([int(i) for i in array_shape])
            return np_array
        except:
            print("shape doesn't match", txt_path)
            return

def main():
    """
    main function
    """
    print("---- align tools: tengine and onnx ----\n")
    args = parse_args()

    print("input onnx model      : %s" % (args.m))
    print("is save result        : %s" % (args.s))
    print("save text path        : %s" % (args.sp))
    print("is align by layer     : %s" % (args.a))
    print("tengine output path   : %s" % (args.to))
    print("tengine model         : %s" % (args.tm))

    align = AlignOnnx(args.m, args.s, args.sp, args.a, args.to, args.tm)
    align()


if __name__ == '__main__':
    main()
