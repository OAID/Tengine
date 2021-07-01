# coding: utf-8
"""Information about Tengine."""

from .base import _LIB, check_call, c_str
import ctypes

class Device(object):
    def __init__(self, driver_name=None, dev_name=None):
        """
        create device, for predefined device but driver does not auto probed device
        :param driver_name: <str> the driver name
        :param dev_name: <str> the device name
        """
        check_call(_LIB.create_device(c_str(driver_name), c_str(dev_name)))
        self.driver = [driver_name, dev_name]
        pass

    def __del__(self):
        """
        destroy device, for predefined device but driver does not auto probed device.
        :return:
        """
        if self.driver:
            check_call(_LIB.destroy_device(*self.driver))
        self.driver = None

    def getAttr(self, item,size):
        buf = ctypes.create_string_buffer('',size=size)
        check_call(_LIB.get_device_attr(c_str(self.driver[1]),c_str(item),ctypes.cast(buf,ctypes.POINTER(ctypes.c_char)),size))
        return buf[:size]

    def setAttr(self,item,obj):
        if type(obj) is int:
            buf = (ctypes.c_int * 1)(obj)
            check_call(_LIB.set_device_attr(c_str(self.driver[1]),c_str(item),ctypes.cast(buf,ctypes.POINTER(ctypes.c_int)),ctypes.sizeof(ctypes.c_int)))
        elif type(obj) is float:
            buf = (ctypes.c_float * 1)(obj)
            check_call(_LIB.set_device_attr(c_str(self.driver[1]),c_str(item),ctypes.cast(buf,ctypes.POINTER(ctypes.c_int)),ctypes.sizeof(ctypes.c_float)))
        elif type(obj) is str:
            buf = ctypes.create_string_buffer(obj,len(obj))
            check_call(
                _LIB.set_device_attr(c_str(self.driver[1]), c_str(item), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)),
                                     ctypes.sizeof(ctypes.c_char)*len(obj)))
        elif type(obj) is list:
            if type(obj[0]) is int:
                buf = (ctypes.c_int * len(obj))(*obj)
                check_call(_LIB.set_device_attr(c_str(self.driver[1]), c_str(item),
                                                ctypes.cast(buf, ctypes.POINTER(ctypes.c_int)),
                                                ctypes.sizeof(ctypes.c_int)*len(obj)))
            elif type(obj[0]) is float:
                buf = (ctypes.c_float*len(obj))(*obj)
                check_call(_LIB.set_device_attr(c_str(self.driver[1]), c_str(item),
                                                ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)),
                                                ctypes.sizeof(ctypes.c_float)*len(obj)))
        else:
            print("not support type: {} yet.".format(type(obj)))

    @property
    def policy(self):
        """
        get the device working mode.
        :return: <int> >= 0 : the mode, -1: fail.
        """
        return _LIB.get_device_policy(c_str(self.driver[1]))

    @policy.setter
    def policy(self, policy):
        """
        set the device working policy.
        :param policy:
        :return:
        """
        return _LIB.set_device_policy(c_str(self.driver[1]), policy)
