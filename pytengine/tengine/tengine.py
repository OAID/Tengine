# coding: utf-8
"""Information about Tengine."""
import ctypes
from .base import _LIB
from .base import check_call, c_str, log_print_t, cpu_info
from .graph import Graph
from .context import Context
from .node import Node
from .tensor import Tensor
from .device import Device
import os

class Tengine(object):
    """initialize the tengine"""

    def __init__(self):
        """
        Initialize the tengine, only can be called once.
        """
        check_call(_LIB.init_tengine())
        self.log_lever = property(None, self.__log_lever)
        self.log_print = property(None, self.__log_output)
        self.plugin = self.Plugin()

    def __del__(self):
        """
        Release the tengine, only can be called once.
        :return: None
        """
        print("release tengine")
        _LIB.release_tengine()

    @property
    def __errno__(self):
        """
        return the error number
        list of the symbolic error name follows glibc definitions
        :return: <int> error number
        """
        return _LIB.get_tengine_errno()

    @property
    def __version__(self):
        """
        Get the version of the tengine.
        :return: <str>
        """
        _LIB.get_tengine_version.restype = ctypes.c_char_p
        return _LIB.get_tengine_version()

    def requestTengineVersion(self, version):
        """
        Check the run-time library supports the verson.
        :param version: <str> version: A c string returned by tg.__version__
        :return: 1: support, 0: not support
        """
        return _LIB.request_tengine_version(c_str(version))

    def setDefaultDevice(self,dev_name):
        """
        set The default device.
        :param dev_name: <str> The device name.
        :return: None
        """
        check_call(_LIB.set_default_device(c_str(dev_name)))

    def __log_lever(self, value):
        """
        Set the logger level.
        :param value: <log_level> The log level. like: tg.LOG_EMERG, tg.LOG_ALERT, etc.
        :return: None
        """
        self.log_level = value
        _LIB.set_log_level(ctypes.c_int(value))

    def __log_output(self, func):
        """
        set the print function of log.
        :param func: python function like: def function(str): ...
        :return: None
        """
        _LIB.set_log_output(log_print_t(func))

    def get_predefined_cpu(self,name):
        _LIB.get_predefined_cpu.restype = ctypes.POINTER(cpu_info)
        return _LIB.get_predefined_cpu(c_str(name))[0]

    def set_online_cpu(self,cpu_info,cpu_list):
        cpu_list = (ctypes.c_int * len(cpu_list))(*cpu_list)
        return _LIB.set_online_cpu(ctypes.byref(cpu_info), ctypes.byref(cpu_list), len(cpu_list))

    def create_cpu_device(self,name,cpu_info):
        return _LIB.create_cpu_device(c_str(name), ctypes.byref(cpu_info))

    class Plugin(object):
        def __init__(self):
            self.pluginList = []
            pass

        def add(self, name, fname, init_func):
            """
            Load one plugin from disk, and execute the init function.
            :param name:  <str> plugin_name: Plugin name.
            :param fname: <str> fname: Plugin file name.
            :param init_func: The name of the init function.
            :return: None
            """
            self.pluginList.append(name)
            check_call(_LIB.load_tengine_plugin(c_str(name), c_str(fname), c_str(init_func)))

        def remove(self, name, del_func):
            """
            Unload one plugin and call the release function.
            :param name: <str> plugin_name: The name of plugin.
            :param del_func: <str> rel_func_name: The release function name.
            :return: None
            """
            self.pluginList.remove(name)
            check_call(_LIB.unload_tengine_plugin(c_str(name), c_str(del_func)))

        def __len__(self):
            """
            Get the number of loaded plugin.
            :return: <int> The plugin number.
            """
            return _LIB.get_tengine_plugin_number()

        def __getitem__(self, item):
            """
            Get the name of idx plugin.
            :param item: <int> The index of loaded plugin.
            :return: The name of plugin.
            """
            if type(item) is int:
                _LIB.get_tengine_plugin_name.restype = ctypes.c_char_p
                return _LIB.get_tengine_plugin_name(item)
            else:
                return None


tg = Tengine()
tg.Graph = Graph
tg.Context = Context
tg.Node = Node
tg.Tensor = Tensor
tg.Device = Device


# /* the data type of the tensor */
(tg.TENGINE_DT_FP32,
  tg.TENGINE_DT_FP16,
  tg.TENGINE_DT_INT8,
  tg.TENGINE_DT_UINT8,
  tg.TENGINE_DT_INT32,
  tg.TENGINE_DT_INT16) = map(int,range(6))


# /* layout type, not real layout */
(tg.TENGINE_LAYOUT_NCHW,
 tg.TENGINE_LAYOUT_NHWC) = map(int, range(2))

# /* tensor type: the content changed or not during inference */
(tg.TENSOR_TYPE_UNKNOWN,
 tg.TENSOR_TYPE_VAR,
 tg.TENSOR_TYPE_CONST,
 tg.TENSOR_TYPE_INPUT,
 tg.TENSOR_TYPE_DEP) = map(int, range(5))

# /* node dump action definition */
(tg.NODE_DUMP_ACTION_DISABLE,
 tg.NODE_DUMP_ACTION_ENABLE,
 tg.NODE_DUMP_ACTION_START,
 tg.NODE_DUMP_ACTION_STOP,
 tg.NODE_DUMP_ACTION_GET) = map(int, range(5))

# /* graph perf action definition */
(tg.GRAPH_PERF_STAT_DISABLE,
 tg.GRAPH_PERF_STAT_ENABLE,
 tg.GRAPH_PERF_STAT_STOP,
 tg.GRAPH_PERF_STAT_START,
 tg.GRAPH_PERF_STAT_RESET,
 tg.GRAPH_PERF_STAT_GET) = map(int, range(6))

# /* quant mode */
(tg.TENGINE_QUANT_FP16,
 tg.TENGINE_QUANT_INT8,
 tg.TENGINE_QUANT_UINT8) = map(int, range(3))

# /*follow the std. UNIX log level definitioin */
(tg.LOG_EMERG,
 tg.LOG_ALERT,
 tg.LOG_CRIT,
 tg.LOG_ERR,
 tg.LOG_WARNING,
 tg.LOG_NOTICE,
 tg.LOG_INFO,
 tg.LOG_DEBUG
) = map(int,range(8))

# * todo: should add suspend? */
(
    tg.GRAPH_STAT_CREATED,
    tg.GRAPH_STAT_READY,
    tg.GRAPH_STAT_RUNNING,
    tg.GRAPH_STAT_DONE,
    tg.GRAPH_STAT_ERROR
) = map(int,range(5))

# /* graph_exec_event */
(
    tg.GRAPH_EXEC_START,
    tg.GRAPH_EXEC_SUSPEND,
    tg.GRAPH_EXEC_RESUME,
    tg.GRAPH_EXEC_ABORT,
    tg.GRAPH_EXEC_DONE
) = map(int,range(5))

# /* device_policy */
(
    tg.DEFAULT_POLICY,
    tg.LATENCY_POLICY,
    tg.LOW_POWER_POLICY
) = map(int,range(3))


