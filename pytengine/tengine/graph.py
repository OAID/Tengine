# coding: utf-8
"""Information about Tengine."""

import ctypes
import numpy as np
from .base import _LIB, c_str, graph_t, node_t, tensor_t, Status, perf_info, check_call, event_handler_t
from .tensor import Tensor
from .node import Node
import time

class Graph(object):
    def __init__(self, context=None, model=None, *kwarg):
        """
        create the run-time graph for execution from a saved model
        if model is None, an empty graph will be created
        :param context: <context object> or None. the context for this graph to run inside
        :param model: <str> model format : caffe , mxnet , tengine , ...
        :param kwarg: the path of the model file
        """
        _LIB.create_graph.restype = graph_t
        if context:
            context = context.context
        if model:
            params = [ c_str(item) for item in kwarg]
            self.graph = _LIB.create_graph(ctypes.c_void_p(context), c_str(model), *params)
        else:
            self.graph = _LIB.create_graph(ctypes.c_void_p(context), None)
        self.attr = {}
        pass

    def __del__(self):
        """
        destory the run-time graph and release allocated resource.
        :return:
        """
        if self.graph:
            _LIB.destroy_graph(ctypes.c_void_p(self.graph))
            print("release graph")
        self.graph = None
        pass

    def __getitem__(self, idx):
        """
        get the output node by idx
        :param idx: <int> like: 0,1,2,...
        :return: <node object> output node object
        """
        return self.getNodeByIdx(idx)

    def setEvent(self,event,cb_func,cb_arg):
        """
        set the event hook for graph execution
        :param event: the event to be hooked
        :param cb_func: the callback function
        :param cb_arg: the argument will be passed to callback function
        :return: None
        """
        check_call(_LIB.set_graph_event_hook(ctypes.c_void_p(self.graph),event,event_handler_t(cb_func),ctypes.pointer(cb_arg)))

    def save(self, model, *kwarg):
        """
        save the graph into file using the model format
        :param model: <str> model format
        :param kwarg: <str> save file path
        :return: 0: success, -1: fail
        """
        params = [c_str(item) for item in kwarg]
        return _LIB.save_graph(ctypes.c_void_p(self.graph), c_str(model), *params)

    def quant(self, mode, idxs):
        """
        quant the graph according to the quant mode
        :param mode: <quant_mode> like: tg.TENGINE_QUANT_FP16 etc.
        :param idxs: <list> list of nodes not quant
        :return:
        """
        _LIB.quant_graph.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int]
        return _LIB.quant_graph(ctypes.c_void_p(self.graph), mode, np.ctypeslib.as_ctypes(idxs), len(idxs))

    def setlayout(self, type):
        """
        set the layer type of the graph
        :param type: <layout_type> like: tg.TENGINE_LAYOUT_NCHW, tg.TENGINE_LAYOUT_NHWC
        :return: 0: success, -1: fail
        """
        return _LIB.set_graph_layout(ctypes.c_void_p(self.graph), type)

    def setNode(self, input_nodes=[], output_nodes=[]):
        """
        designate the input nodes and output nodes of the graph
        :param input_nodes: <list> the node name list of input nodes
        :param output_nodes: <list> the node name list of output nodes
        :return:
        """
        if len(input_nodes) != 0:
            num = len(input_nodes)
            c_arr_buf = ctypes.c_char_p * num
            input_ = [c_str(item) for item in input_nodes]
            c_arr = c_arr_buf(*input_)
            check_call(_LIB.set_graph_input_node(ctypes.c_void_p(self.graph), c_arr, num))
            pass
        if len(output_nodes) != 0:
            num = len(output_nodes)
            c_arr_buf = ctypes.c_char_p * num
            output_ = [c_str(item) for item in output_nodes]
            c_arr = c_arr_buf(*output_)
            check_call(_LIB.set_graph_output_node(ctypes.c_void_p(self.graph), c_arr, num))
            pass

    def __add__(self, *other):
        """
        merge several graph into this graph, important: all graphs should be in the same context
        :param other: <graph objects>
        :return:
        """
        _LIB.merge_graph.restype = graph_t
        c_graph = [ctypes.c_void_p(p) for p in other]
        graph = _LIB.merge_graph(ctypes.c_void_p(self.graph), *c_graph)
        return Graph(graph=graph)

    def getInputNodeNumber(self):
        """
        get the number of input nodes of the graph
        :return: <int> number of input nodes
        """
        return _LIB.get_graph_input_node_number(self.graph)

    def getInputNodeByIdx(self, idx):
        """
        get the node object by the index of input node of this graph
        :param idx: <idx> index of the input nodes
        :return: <node object>
        """
        _LIB.get_graph_input_node = node_t
        node = _LIB.get_graph_input_node(ctypes.c_void_p(self.graph), idx)
        return Node(node=node)

    def getOutputNodeNumber(self):
        """
        get the number of the output nodes of this graph
        :return: number of the output nodes
        """
        return _LIB.get_graph_output_node_number(ctypes.c_void_p(self.graph))

    def getOutputNodeByIdx(self, idx):
        """
        get the node object by the index of the output node of this graph
        :param idx: <int> index of the output node
        :return: <node object>
        """
        _LIB.get_graph_output_node.restype = node_t
        node = _LIB.get_graph_output_node(ctypes.c_void_p(self.graph), idx)
        return Node(node=node)

    def getOutputTensor(self, nodeidx, idx):
        """
        get the tensor object of a graph output node
        :param nodeidx: <int> the output node index.
        :param idx: <int> the output tensor index of the output node.
        :return: <tensor object>
        """
        _LIB.get_graph_output_tensor.restype = tensor_t
        tensor = _LIB.get_graph_output_tensor(ctypes.c_void_p(self.graph), nodeidx, idx)
        return Tensor(tensor=tensor)

    def getInputTensor(self, nodeidx, idx):
        """
        get a tensor object of this graph input tensor
        :param nodeidx: <int> input node index
        :param idx:  <int> the output tensor index of the input node
        :return:
        """
        _LIB.get_graph_input_tensor.restype = tensor_t
        tensor = _LIB.get_graph_input_tensor(ctypes.c_void_p(self.graph), nodeidx, idx)
        return Tensor(tensor=tensor)

    def getNodeByName(self, name):
        """
        get the node object of the graph.
        :param name: <str> the node name
        :return: <node object>
        """
        _LIB.get_graph_node.restype = node_t
        node = _LIB.get_graph_node(ctypes.c_void_p(self.graph), c_str(name))
        return Node(node=node)

    def getNodeNumber(self):
        """
        get graph node number.
        :return: >=0: the number of the graph node, -1: on error
        """
        return _LIB.get_graph_node_number(ctypes.c_void_p(self.graph))

    def getNodeByIdx(self, idx):
        """
        get graph node by index
        :param idx: <int> the node index
        :return: <node object>
        """
        _LIB.get_graph_node_by_idx = node_t
        node = _LIB.get_graph_node_by_idx(ctypes.c_void_p(self.graph), idx)
        return Node(node=node)

    def getTensorByName(self, name):
        """
        get a tensor object by tensor name
        :param name: <str> name of the tensor
        :return: <tensor object>
        """
        _LIB.get_graph_tensor.restype = tensor_t
        tensor = _LIB.get_graph_tensor(ctypes.c_void_p(self.graph), c_str(name))
        return Tensor(tensor=tensor)

    def setAttr(self, attr_name, obj):
        """
        to set proprietary attribute items for graph, for the backend device to run
        :param attr_name:<str> the attribute name
        :param obj: <int> or <float> or <str> attribute value
        :return: 0: success, -1: fail.
        """
        if type(obj) == type(0):
            i_inp = (ctypes.c_int*1)(obj)
            self.attr[attr_name] = {'type': type(obj), 'len': ctypes.sizeof(i_inp)}
            return _LIB.set_graph_attr(ctypes.c_void_p(self.graph), c_str(attr_name), ctypes.cast(i_inp,ctypes.POINTER(ctypes.c_int)),
                                ctypes.sizeof(i_inp))
        elif type(obj) == type(0.1):
            f_inp = (ctypes.c_float*1)(obj)
            self.attr[attr_name] = {'type': type(obj), 'len': ctypes.sizeof(f_inp)}
            return _LIB.set_graph_attr(ctypes.c_void_p(self.graph), c_str(attr_name), ctypes.cast(f_inp,ctypes.POINTER(ctypes.c_float)),
                                ctypes.sizeof(f_inp))
        elif type(obj) == type(""):
            self.attr[attr_name] = {'type': type(obj), 'len': ctypes.sizeof(ctypes.c_char('a')) * len(obj)}
            buf = ctypes.create_string_buffer(obj)
            return _LIB.set_graph_attr(ctypes.c_void_p(self.graph), c_str(attr_name), buf,
                                ctypes.sizeof(ctypes.c_char('a')) * len(obj))
        pass

    def getAttr(self, attr_name):
        """
        get proprietary config items for graph. it is probabaly the config will be passed to the DLA driver.
        :param attr_name: <str> the attribute name
        :return: <int> or <float> or <str>
        """
        if self.attr[attr_name]['type'] == type(0):
            buf = (ctypes.c_int*1)(0)
            _LIB.get_graph_attr(ctypes.c_void_p(self.graph), c_str(attr_name), ctypes.byref(buf),
                                self.attr[attr_name]['len'])
            return buf
        elif self.attr[attr_name]['type'] == type(0.1):
            buf = (ctypes.c_float*1)(0.0)
            _LIB.get_graph_attr(ctypes.c_void_p(self.graph), c_str(attr_name), ctypes.cast(buf,ctypes.POINTER(ctypes.c_float)),
                                self.attr[attr_name]['len'])
            return buf
        elif self.attr[attr_name]['type'] == type("a"):
            buf = ctypes.create_string_buffer("",self.attr[attr_name]['len'])
            _LIB.get_graph_attr(ctypes.c_void_p(self.graph), c_str(attr_name), ctypes.byref(buf),
                                self.attr[attr_name]['len'])
            return buf[:]
        else:
            return None

    def setGdMethod(self, methor,*params):
        """
        set the gradient descent method
        :param methor: <int>[<int> ...] the gradient descent.
        :return: 0: success, -1 fail.
        """
        check_call(_LIB.set_graph_gd_method(ctypes.c_void_p(self.graph), methor,*params))

    def preRun(self):
        """
        Initialize resource for graph execution
        :return: None
        """
        check_call(_LIB.prerun_graph(ctypes.c_void_p(self.graph)))

    def run(self, block=0):
        """
        execute graph
        :param block: 0: no_blocking,1 : blocking
        :return: None
        """
        check_call(_LIB.run_graph(ctypes.c_void_p(self.graph), block))

    def wait(self, try_wait=0):
        """
        wait graph execution done
        :param try_wait: <bool> if set, just check status and return
        :return: 1: graph is done, 0: try again
        """
        return _LIB.wait_graph(ctypes.c_void_p(self.graph), try_wait)

    def postRun(self):
        """
        release the resource for graph execution.
        :return:None
        """
        check_call( _LIB.postrun_graph(ctypes.c_void_p(self.graph)))

    @property
    def __status__(self):
        """
        get the status of graph execution
        :return: <status>
        """
        ret = _LIB.get_graph_exec_status(ctypes.c_void_p(self.graph))
        return Status(ret)

    def setDevice(self, dev_name):
        """
        set the device to execution a graph
        :param dev_name: <str> The device name to run the node.
        :return: 0 : success, <0 : error
        """
        return _LIB.set_graph_device(ctypes.c_void_p(self.graph), c_str(dev_name))

    def doPerfStat(self, action):
        """
        start or stop the perf stats
        :param action: <int> 0 stop, 1 start, 2 reset counter
        :return: 0: success, -1: fail
        """
        return _LIB.do_graph_perf_stat(ctypes.c_void_p(self.graph), action)

    def getPerfStat(self, size):
        """
        get graph performance stats records
        If the returned number equals buf_size, there may be som records do not be
        retrieved yet
        :param size: <int> the number of record pointer may be stored in buf
        :return: <perf_info list>
        """
        info = (perf_info*size)(None)
        _LIB.get_graph_perf_stat(ctypes.c_void_p(self.graph), ctypes.byref(info), size)
        return info

    def dump(self):
        """
        Dump the run-time graph.
        Note: If the graph is dumpped after prerun(), it will dump the optimized graph instead of the origin one.
        :return: None
        """
        check_call(_LIB.dump_graph(self.graph))

    def set_input(self,image,shape,input_tensor,mean=[0,0,0],scale=1):
        """
        set graph input data and shape here. it will do some simple pre-treatment here
        :param image: <list> or <ndarray> : input data
        :param shape: <list> shape
        :param input_tensor: one input tensor of this graph to set
        :param mean: <list> mean value
        :param scale: <float> scale value
        :return: None
        """

        data = image
        if type(data) is np.ndarray:
            data = data.swapaxes(1,2)
            data = data.swapaxes(0,1)
            data = data.flatten()
        input_tensor.shape = shape
        input_tensor.buf = data
        pass
