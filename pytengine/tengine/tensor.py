# coding: utf-8
"""Information about Tengine."""

import ctypes
from .base import _LIB, c_str, tensor_t, ctypes2numpy_shared, ctypes2buffer, DType, check_call
import numpy as np
import time

class Tensor(object):
    def __init__(self, graph=None, name=None, type=None,tensor=None):
        """
        create a tensor object by tensor name or by tensor pointer
        :param graph: <graph object>
        :param name: <str> tensor name
        :param type: <data_type> : the data type
        :param tensor: <tensor pointer> normal used by the sys
        """
        if tensor:
            self.tensor = tensor
        else:
            _LIB.create_graph_tensor.restype = tensor_t
            self.tensor = _LIB.create_graph_tensor(ctypes.c_void_p(graph.graph), c_str(name), type)
        self._data = None
        self._buf = None
        self.shape_number = None
        self.length = 4
        pass

    def __del__(self):
        """
        release the tensor object
        :return: None
        """
        if self.tensor:
            _LIB.release_graph_tensor(ctypes.c_void_p(self.tensor))
        self.tensor = None

    @property
    def __name__(self):
        """
        get the name of the tensor object
        :return: <str> name of the tensor object
        """
        _LIB.get_tensor_name.restype = ctypes.c_char_p
        return _LIB.get_tensor_name(ctypes.c_void_p(self.tensor))

    @property
    def __len__(self):
        """
        get the byte size of a tensor should occupy
        :return: <int> 0: the shape of the tensor is not set yet.
        """
        return _LIB.get_tensor_buffer_size(ctypes.c_void_p(self.tensor))

    @property
    def shape(self):
        """
        get the shape of tensor
        :return: <list> An int array
        """
        dim = (ctypes.c_int * self.length)(0, 0, 0, 0)
        _LIB.get_tensor_shape.restype = ctypes.c_int
        self.shape_number = _LIB.get_tensor_shape(ctypes.c_void_p(self.tensor), ctypes.cast(dim, ctypes.POINTER(ctypes.c_int)), 4)
        return ctypes2numpy_shared(ctypes.cast(dim, ctypes.POINTER(ctypes.c_int)), (1, 4))[0]

    @property
    def shape_num(self):
        """
        valid dim number
        :return: >=1 the dim number, -1: fail
        """
        return self.shape_number

    @shape.setter
    def shape(self, dims):
        """
        set the shape of tensor
        :param dims: <list> An int array to represent shape
        :return: None
        """
        self.length = len(dims)
        c_dims = (ctypes.c_int * self.length)(*dims)
        check_call(_LIB.set_tensor_shape(ctypes.c_void_p(self.tensor), ctypes.cast(c_dims, ctypes.POINTER(ctypes.c_int)), self.length))

    def getbuffer(self, type=int):
        """
        Get the byte size of a tensor should occupy.
        :param type: <type> int or float
        :return:
        """
        if type is int:
            _LIB.get_tensor_buffer.restype = ctypes.POINTER(ctypes.c_int)
        elif type is float:
            _LIB.get_tensor_buffer.restype = ctypes.POINTER(ctypes.c_float)
        elif type is str:
            _LIB.get_tensor_buffer.restype = ctypes.POINTER(ctypes.c_char)
        else:
            return None
        return _LIB.get_tensor_buffer(ctypes.c_void_p(self.tensor))

    @property
    def buf(self):
        """
        Get the byte size of a tensor should occupy. only for buf that user set themselves
        :return: <list> data list
        """
        if not self._buf:
            return None
        if self._buf[0] is ctypes.c_char:
            _LIB.get_tensor_buffer.restype = ctypes.POINTER(self._buf[0])
            res = _LIB.get_tensor_buffer(ctypes.c_void_p(self.tensor))
            return res[:self._buf[1]]
        _LIB.get_tensor_buffer.restype = ctypes.POINTER(self._buf[0])
        res = _LIB.get_tensor_buffer(ctypes.c_void_p(self.tensor))
        return res[:self._buf[1]]

    @buf.setter
    def buf(self, value):
        """
        Set the buffer of the tensor.
        :param value: <int list> or <float list> or <str list>
        :return: None
        """
        _LIB.set_tensor_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        check_call(_LIB.set_tensor_buffer(self.tensor, np.ctypeslib.as_ctypes(value), len(value) * ctypes.sizeof((ctypes.c_float))))

    def getData(self):
        """
        get tensor data.
        :return: <list>
        """
        if self._data[0] == ctypes.c_int:
            c_data = (ctypes.c_int * self._data[1])(0)
            check_call(_LIB.get_tensor_data(ctypes.c_void_p(self.tensor), ctypes.pointer(c_data), ctypes.sizeof(ctypes.c_int)))
            return ctypes2numpy_shared(ctypes.cast(c_data, ctypes.POINTER(ctypes.c_int)), (1, self._data[1]))
        elif self._data[0] == ctypes.c_float:
            c_data = (ctypes.c_float * self._data[1])(0.0)
            check_call(_LIB.get_tensor_data(ctypes.c_void_p(self.tensor), ctypes.cast(c_data, ctypes.c_char_p),
                                 ctypes.sizeof(ctypes.c_float)))
            return c_data[:self._data[1]]
        elif self._data[0] == ctypes.c_char:
            c_data = ctypes.create_string_buffer("", self._data[1])
            check_call(_LIB.get_tensor_data(ctypes.c_void_p(self.tensor), ctypes.cast(c_data, ctypes.c_void_p),
                                 ctypes.sizeof(ctypes.c_char) * self._data[1]))
            return ctypes2buffer(ctypes.cast(c_data, ctypes.POINTER(ctypes.c_char)), self._data[1])
        else:
            return None

    def setData(self, data):
        """
        Copy the data to tensor buffer.
        :param data: <int> or <float> or <str> or <list>  the input data
        :return: None
        """
        if type(data) == type(0):
            self._data = [ctypes.c_int, 1]
            c_data = (ctypes.c_int * 1)(data)
            check_call(_LIB.set_tensor_data(ctypes.c_void_p(self.tensor), ctypes.cast(c_data, ctypes.POINTER(ctypes.c_int)),
                                        ctypes.sizeof(ctypes.c_int)))
        elif type(data) == type(0.1):
            self._data = [ctypes.c_float, 1]
            c_data = (ctypes.c_float * 1)(data)
            check_call(_LIB.set_tensor_data(ctypes.c_void_p(self.tensor),
                                        ctypes.cast(c_data, ctypes.POINTER(ctypes.c_float)),
                                        ctypes.sizeof(ctypes.c_float)))
        elif type(data) == type("0"):
            self._data = [ctypes.c_char, len(data)]
            c_data = ctypes.create_string_buffer(data)
            check_call(_LIB.set_tensor_data(ctypes.c_void_p(self.tensor), ctypes.cast(c_data, ctypes.c_char_p),
                                        ctypes.sizeof(c_data)))
        elif type(data) == type([]):
            size = len(data)
            if size:
                if type(data[0]) == type(0):
                    self._data = [ctypes.c_int, size]
                    c_data = (ctypes.c_int * size)(data)
                    check_call(_LIB.set_tensor_data(ctypes.c_void_p(self.tensor),
                                                ctypes.cast(c_data, ctypes.POINTER(ctypes.c_int)),
                                                ctypes.sizeof(ctypes.c_int) * size))
                elif type(data[0]) == type(0.0):
                    self._data = [ctypes.c_float, size]
                    c_data = (ctypes.c_float * size)(data)
                    check_call(_LIB.set_tensor_data(ctypes.c_void_p(self.tensor),
                                                ctypes.cast(c_data, ctypes.POINTER(ctypes.c_float)),
                                                ctypes.sizeof(ctypes.c_float) * size))
                else:
                    return -1
            else:
                return -1
        else:
            return -1

    @property
    def dtype(self):
        """
        Get the data type of the tensor.
        :return: <The tensor type> like: TENGINE_DT_FP32 etc, -1 on error.
        """
        d =  _LIB.get_tensor_data_type(ctypes.c_void_p(self.tensor))
        return DType(d)

    @dtype.setter
    def dtype(self, type):
        """
        Get the data type of the tensor.
        :param type:
        :return: None
        """
        check_call(_LIB.set_tensor_data_type(ctypes.c_void_p(self.tensor), type))

    def setQuantParam(self, scale, zero_point, number):
        """
        Set tensor quant parameters
        :param scale: <float list> the scale list
        :param zero_point: <int> the zero point address
        :param number: <int> the element number of array
        :return: None
        """
        _LIB.set_tensor_quant_param.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_int),ctypes.c_int]
        check_call(_LIB.set_tensor_quant_param(ctypes.c_void_p(self.tensor), np.ctypeslib.as_ctypes(scale),
                                           np.ctypeslib.as_ctypes(zero_point), number))

    def getQuantParam(self, number):
        """
        Get tensor quant parameters.
        :param number: <int> the element number of array
        :return: (<scale list>,<zero_point list>)
        """
        scale = ctypes.c_float * number
        zero_point = ctypes.c_int * number
        _LIB.get_tensor_quant_param(ctypes.c_void_p(self.tensor), ctypes.POINTER(scale), ctypes.POINTER(zero_point),
                                    number)
        return scale[:number], zero_point[:number]
