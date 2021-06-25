# coding: utf-8
"""Information about Tengine."""
import ctypes
from .base import _LIB,c_str,context_t, check_call


class Context(object):
    def __init__(self, name, empty=True):
        """
        create one execution context with name
        :param name: <str> name of the created context
        :param empty: <bool> True: no device assigned , False: all proved devices will be assigned
        """
        self.__attr__ = {}
        _LIB.create_context.restype = context_t
        self.context = _LIB.create_context(c_str(name), empty)
        pass

    def __del__(self):
        """
        destory and reclaim the resource related with the context
        :return:
        """
        _LIB.destroy_context(ctypes.c_void_p(self.context))

    def getDevNumber(self):
        """
        get the device number assigned to a context
        :return: <int> numbers of the context device
        """
        return _LIB.get_context_device_number(ctypes.c_void_p(self.context))

    def getDevByIdx(self, idx):
        """
        get the name of the idx device in a context
        :param idx: <int> 0,1,...,
        :return <str> name of the device

        """
        _LIB.get_context_device_name.restype = ctypes.c_char_p
        return _LIB.get_context_device_name(ctypes.c_void_p(self.context), idx)

    def addDev(self,dev_name):
        """
        add a device into one context
        :param dev_name: <str> name of the device (created by user)
        :return: <int> 0: success , -1 : Fail.
        """
        check_call(_LIB.add_context_device(ctypes.c_void_p(self.context), c_str(dev_name)))

    def rmDev(self,dev_name):
        """
        remove a device from one context
        :param dev_name: <str> name of the device (created by user)
        :return: <int> 0: success , -1 : Fail.
        """
        return _LIB.remove_context_device(ctypes.c_void_p(self.context), c_str(dev_name))

    def setAttr(self,attr,obj):
        """
        set attribute item of a context
        :param attr: <str> attr_name, the attribute item name.
        :param obj: <int> or <float> or <str> or <list> the data want to set
        :return: None
        """
        if type(obj) is int:
            buf = (ctypes.c_int * 1)(obj)
            check_call(_LIB.set_context_attr(ctypes.c_void_p(self.context),c_str(attr),ctypes.cast(buf,ctypes.POINTER(ctypes.c_int)),ctypes.sizeof(ctypes.c_int)))
        elif type(obj) is float:
            buf = (ctypes.c_float * 1)(obj)
            check_call(_LIB.set_context_attr(ctypes.c_void_p(self.context), c_str(attr),
                                             ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)),
                                             ctypes.sizeof(ctypes.c_float)))
        elif type(obj) is str:
            buf = ctypes.create_string_buffer(obj,len(obj))
            check_call(_LIB.set_context_attr(ctypes.c_void_p(self.context), c_str(attr),
                                             ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)),
                                             ctypes.sizeof(ctypes.c_char)*len(obj)))
        elif type(obj) is list:
            if type(obj[0]) is int:
                buf = (ctypes.c_int * len(obj))(*obj)
                check_call(_LIB.set_context_attr(ctypes.c_void_p(self.context), c_str(attr),
                                                 ctypes.cast(buf, ctypes.POINTER(ctypes.c_int)),
                                                 ctypes.sizeof(ctypes.c_int)*len(obj)))
            elif type(obj[0]) is float:
                buf = (ctypes.c_float * len(obj))(*obj)
                check_call(_LIB.set_context_attr(ctypes.c_void_p(self.context), c_str(attr),
                                                 ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)),
                                                 ctypes.sizeof(ctypes.c_float)*len(obj)))
        else:
            print("not surpported type: {} yet.".format(type(obj)))

    def getAttr(self,attr_name,size):
        """
        get the attribute item of a context
        :param attr_name: <str> the attribute item name
        :param size: <int> the buffer size
        :return: data buffer
        """
        buf = ctypes.create_string_buffer('',size=size)
        check_call(_LIB.get_context_attr(ctypes.c_void_p(self.context),c_str(attr_name)),ctypes.cast(buf,ctypes.POINTER(ctypes.c_char),size))

