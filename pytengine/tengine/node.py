# coding: utf-8
"""Information about Tengine."""
import ctypes
from .base import _LIB,  c_str, node_t, tensor_t, check_call
from .tensor import Tensor
from .base import tensor_dump_header


class Node(object):
    def __init__(self, graph = None,name=None,op=None,node=None):
        """
        Create a node object for the graph.
        :param graph: <graph object>
        :param name: <str> node_name: The name of the node.
        :param op: <str> op_name: The name of the operate.
        :param node: <node pointer>
        """
        if node:
            self.node = node
        else:
            _LIB.create_graph_node.restype = node_t
            self.node = _LIB.create_graph_node(ctypes.c_void_p(graph.graph), c_str(name), c_str(op))
        self._attr = {}
        pass

    def __del__(self):
        """
        release this node
        :return: None
        """
        if self.node:
            _LIB.release_graph_node(ctypes.c_void_p(self.node))
        self.node = None

    @property
    def __name__(self):
        """
        Get the node name.
        :return: The node name, None on error.
        """
        _LIB.get_node_name.restype = ctypes.c_char_p
        return _LIB.get_node_name(ctypes.c_void_p(self.node))

    @property
    def __op__(self):
        """
        Get the node op.
        :return: The op name, None on error.
        """
        _LIB.get_node_op.restype = ctypes.c_char_p
        return _LIB.get_node_op(ctypes.c_void_p(self.node))

    def getInputTensorByIdx(self, idx):
        """
        Get the input tensor handle of a node.
        :param idx: <int> The index of the input tensor.
        :return: The tensor name or None on error
        """
        _LIB.get_node_input_tensor.restype = tensor_t
        tensor =  _LIB.get_node_input_tensor(ctypes.c_void_p(self.node), idx)
        return Tensor(tensor=tensor)

    def getOutputTensorByIdx(self, idx):
        """
        Get the output tensor handle of a node.
        :param idx: <int> The index of the output tensor.
        :return: The tensor handle or None on error.
        """
        _LIB.get_node_output_tensor.restype = tensor_t
        tensor = _LIB.get_node_output_tensor(ctypes.c_void_p(self.node), idx)
        return Tensor(tensor = tensor)

    def setInputTensorByIdx(self, idx, tensor):
        """
        Set a node's the idx input tensor.
        :param idx: <int> The index of the input tensor.
        :param tensor: <tensor object>
        :return: None
        """
        check_call(_LIB.set_node_input_tensor(ctypes.c_void_p(self.node), idx, ctypes.c_void_p(tensor.tensor)))

    def setOutputTensorByIdx(self, idx, tensor, type):
        """
        Set a node's the #idx output tensor.
        :param idx: <int> tensor index of the output tensor
        :param tensor: <tensor object>
        :param type: <tensor_type> like: tg.TENSOR_TYPE_UNKNOWN,tg.TENSOR_TYPE_VAR etc.
        :return:None
        """
        check_call(_LIB.set_node_output_tensor(ctypes.c_void_p(self.node), idx, ctypes.c_void_p(tensor.tensor), type))

    def getOutputNumber(self):
        """
        Get the output tensor number of a node.
        :return: <int> >=1: the number of output tensor, -1: error
        """
        return _LIB.get_node_output_number(ctypes.c_void_p(self.node))

    def getInputNumber(self):
        """
        Get the input tensor number of a node.
        :return: <int> >=1: the number of output tensor, -1: error
        """
        return _LIB.get_node_input_number(ctypes.c_void_p(self.node))

    def setAttr(self, attr_name, attr):
        """
        set the attribute value of a node
        Note: if the attribute is not existed, add first.
        :param attr_name: <str> attribute name
        :param attr: <int> or <float>
        :return: None
        """
        if type(attr) is int:
            data = ctypes.c_int(0)
            self._attr[attr_name] = ctypes.c_int
            ret = _LIB.get_node_attr_int(ctypes.c_void_p(self.node), c_str(attr_name), ctypes.pointer(data))
            if ret < 0:
                _LIB.add_node_attr(ctypes.c_void_p(self.node),c_str(attr_name),c_str("int"),ctypes.sizeof(ctypes.c_int))
            _LIB.set_node_attr_int(ctypes.c_void_p(self.node),c_str(attr_name),ctypes.pointer(ctypes.c_int(attr)))
        elif type(attr) is float:
            self._attr[attr_name] = ctypes.c_float
            data = ctypes.c_float(0.0)
            ret = _LIB.get_node_attr_float(ctypes.c_void_p(self.node), c_str(attr_name), ctypes.pointer(data))
            if ret < 0:
                _LIB.add_node_attr(ctypes.c_void_p(self.node), c_str(attr_name), c_str("float"),
                                   ctypes.sizeof(ctypes.c_int))
            _LIB.set_node_attr_float(ctypes.c_void_p(self.node), c_str(attr_name), ctypes.pointer(ctypes.c_float(attr)))


    def getAttr(self,attr,type=None):
        """
        Get the attribute value (float or int) of a node
        :param attr: <str> The name of the attribute to be retrieval.
        :param type: like: "int" or "float"
        :return: <int> or <float>
        """
        if (self._attr.has_key(attr) and self._attr[attr] is ctypes.c_int) or type is int:
            data = ctypes.c_int(0)
            _LIB.get_node_attr_int(ctypes.c_void_p(self.node),c_str(attr),ctypes.pointer(data))
            return data.value
        elif (self._attr.has_key(attr) and self._attr[attr] is ctypes.c_float) or type is float:
            data = ctypes.c_float(0.0)
            _LIB.get_node_attr_float(ctypes.c_void_p(self.node),c_str(attr),ctypes.pointer(data))
            return data.value

    def setKernel(self, dev_name, kernel_ops):
        """
        Set customer kernel of a node, on a specific device
        Note: the operate in kernel_ops must be the same as node's operate.
        :param dev_name: <str>
        :param kernel_ops: <custom_kernel_ops object>
        :return: None
        """
        check_call(_LIB.set_custom_kernel(ctypes.c_void_p(self.node), c_str(dev_name), ctypes.byref(kernel_ops)))

    def removeKernel(self, dev_name):
        """
        Remove customer kernel of a node, on a specific device.
        :param dev_name:<str>  The kernel works for which device. None means for default device.
        :return: None
        """
        check_call(_LIB.remove_custom_kernel(ctypes.c_void_p(self.node), c_str(dev_name)))

    def setDevice(self, dev_name):
        """
        Set the device to execution a node.
        :param dev_name: <str> The device name to run the node.
        :return: None
        """
        check_call(_LIB.set_node_device(ctypes.c_void_p(self.node), c_str(dev_name)))

    def getDevice(self):
        """
        get the device the node runs on
        :return: <str> or None
        """
        _LIB.get_node_device.restype = ctypes.c_char_p
        return _LIB.get_node_device(ctypes.c_void_p(self.node))

    def dump(self, action):
        """
        Enable dump function pre-defined on device on a node,
        Note: the dump buffer will be returned by getDumpBuf
        :param action: 1: enable, 0: disable
        :return: None
        """
        check_call(_LIB.do_node_dump(ctypes.c_void_p(self.node), action))

    def getDumpBuf(self, size):
        """
        Get the dump buffer pointer generated by target device
        exact meaning of the dump buffer is decided by device.
        :param size: <int> the pointer array size
        :return: <list> buf list
        """
        buf = (ctypes.POINTER(tensor_dump_header) * size)()
        ret = _LIB.get_node_dump_buffer(ctypes.c_void_p(self.node), ctypes.byref(buf), size)
        return buf[:ret]

