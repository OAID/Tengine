# coding: utf-8
# pylint: disable=invalid-name, no-member, trailing-comma-tuple, bad-mcs-classmethod-argument
"""ctypes library of tegine and helper functions."""
from __future__ import absolute_import

import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as np

from . import libinfo

# pylint: disable=pointless-statement
try:
    basestring
    long
except NameError:
    basestring = str
    long = int
# pylint: enable=pointless-statement

integer_types = (int, long, np.int32, np.int64)
numeric_types = (float, int, long, np.generic)
string_types = basestring,
MAX_SHAPE_DIM_NUM = 4

if sys.version_info[0] > 2:
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    py_str = lambda x: x

log_print_t = ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_char_p)
event_handler_t = ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p)

def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    return os.path.join(os.path.expanduser("~"), '.tengine')


def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('TENGINE_HOME', data_dir_default())


class _NullType(object):
    """Placeholder for arguments"""

    def __repr__(self):
        return '_Null'


_Null = _NullType()


class TytengineError(Exception):
    """Error that will be throwed by all tengine functions."""
    pass


class TGCallbackList(ctypes.Structure):
    """Structure that holds Callback information. Passed to CustomOpProp."""
    _fields_ = [
        ('num_callbacks', ctypes.c_int),
        ('callbacks', ctypes.POINTER(ctypes.CFUNCTYPE(ctypes.c_int))),
        ('contexts', ctypes.POINTER(ctypes.c_void_p))
    ]


def _load_lib():
    """Load library by searching possible path."""
    lib_path = libinfo.find_lib_path()
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    # DMatrix functions
    return lib


# version number
__version__ = libinfo.__version__
_LIB = _load_lib()

# type definitions
context_t = ctypes.c_void_p
graph_t = ctypes.c_void_p
node_t = ctypes.c_void_p
tensor_t = ctypes.c_void_p


# ----------------------------
# helper function definition
# ----------------------------
def check_call(ret):
    if ret != 0:
        raise TytengineError(_LIB.get_tengine_errno())


if sys.version_info[0] < 3:
    def c_str(string):
        if not string:
            return None
        return ctypes.c_char_p(string)


    def c_str_array(strings):
        arr = (ctypes.c_char_p * len(strings))()
        arr[:] = strings
        return arr

else:
    def c_str(string):
        if not string:
            return None
        return ctypes.c_char_p(string.encode('utf-8'))


    def c_str_array(strings):
        arr = (ctypes.c_char_p * len(strings))()
        arr[:] = [s.encode('utf-8') for s in strings]
        return arr


def c_array(ctype, values):
    out = (ctype * len(values))()
    out[:] = values
    return out


def c_array_buf(ctype, buf):
    return (ctype * len(buf)).from_buffer(buf)


def c_handle_array(objs):
    arr = (ctypes.c_void_p * len(objs))()
    arr[:] = [o.handle for o in objs]
    return arr


def ctypes2buffer(cptr, length):
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
        raise TypeError('expected char pointer')
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError('memmove failed')
    return res


def ctypes2numpy_shared(cptr, shape, type=ctypes.c_int):
    if not isinstance(cptr, ctypes.POINTER(type)):
        raise RuntimeError('expected float pointer')
    size = 1
    for s in shape:
        size *= s
    dbuffer = (type * size).from_address(ctypes.addressof(cptr.contents))
    if type == ctypes.c_int:
        return np.frombuffer(dbuffer, dtype=np.int32).reshape(shape)
    elif type == ctypes.c_float:
        return np.frombuffer(dbuffer, dtype=np.float32).reshape(shape)


def build_param_doc(arg_names, arg_types, arg_descs, remove_dup=True):
    param_keys = set()
    param_str = []
    for key, type_info, desc in zip(arg_names, arg_types, arg_descs):
        if key in param_keys and remove_dup:
            continue
        if key == 'num_args':
            continue
        param_keys.add(key)
        ret = '%s : %s' % (key, type_info)
        if len(desc) != 0:
            ret += '\n    ' + desc
        param_str.append(ret)
    doc_str = ('Parameters\n' +
               '----------\n' +
               '%s\n')
    doc_str = doc_str % ('\n'.join(param_str))
    return doc_str


Tengine_type = [
    'TENGINE_DT_FP32',
    'TENGINE_DT_FP16',
    'TENGINE_DT_INT8',
    'TENGINE_DT_UINT8',
    'TENGINE_DT_INT32',
    'TENGINE_DT_INT16'
]


class DType:
    def __init__(self, enum):
        self.enum = enum
        pass

    def __str__(self):
        return "<Tegine dtype :%s>" % Tengine_type[self.enum]

    def __repr__(self):
        return "<Tegine dtype :%s>" % Tengine_type[self.enum]


graph_exec_stat = [
    'GRAPH_STAT_CREATED',
    'GRAPH_STAT_READY',
    'GRAPH_STAT_RUNNING',
    'GRAPH_STAT_DONE',
    'GRAPH_STAT_ERROR'
]


class Status:
    def __init__(self, enum):
        self.enum = enum

    def __str__(self):
        return "<graph exec status :%s>" % graph_exec_stat[self.enum]

    def __repr__(self):
        return "<graph exec status  :%s>" % graph_exec_stat[self.enum]


graph_exec_event = [
    'GRAPH_EXEC_START',
    'GRAPH_EXEC_SUSPEND',
    'GRAPH_EXEC_RESUME',
    'GRAPH_EXEC_ABORT',
    'GRAPH_EXEC_DONE'
]


class Event:
    def __init__(self, enum):
        self.enum = enum

    def __str__(self):
        return "<graph exec event :%s>" % graph_exec_event[self.enum]

    def __repr__(self):
        return "<graph exec event  :%s>" % graph_exec_event[self.enum]


device_policy = [
    'DEFAULT_POLICY',
    'LATENCY_POLICY',
    'LOW_POWER_POLICY'
]


class Policy:
    def __init__(self, enum):
        self.enum = enum

    def __str__(self):
        return "<device_policy :%s>" % device_policy[self.enum]

    def __repr__(self):
        return "<device_policy  :%s>" % device_policy[self.enum]


def _notify_shutdown():
    """Notify Tengine about a shutdown."""
    print("release tengine")
    check_call(_LIB.release_tengine())
    pass


atexit.register(_notify_shutdown)


class tensor_dump_header(ctypes.Structure):
    _fields_ = [('elem_size', ctypes.c_int),
                ('elem_number', ctypes.c_int),
                ('dim_number', ctypes.c_int),
                ('dim', ctypes.c_int * 4),
                ('data', ctypes.c_void_p)]


class perf_info(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p), ('dev_name', ctypes.c_char_p),
                ('count', ctypes.c_uint32), ('min', ctypes.c_uint32),
                ('max', ctypes.c_uint32), ('total_time', ctypes.c_uint64),
                ('base', ctypes.c_uint32)]


class custom_kernel_tensor(ctypes.Structure):
    _fields_ = [('dim', (ctypes.c_int * MAX_SHAPE_DIM_NUM)),
                ('dim_num', ctypes.c_int),
                ('element_num', ctypes.c_int),
                ('element_size', ctypes.c_int),
                ('data_type', ctypes.c_int),
                ('dev_type', ctypes.c_int),
                ('layout_type', ctypes.c_int),
                ('quant_type', ctypes.c_int),
                ('scale', ctypes.POINTER(ctypes.c_float)),
                ('zero_point', ctypes.POINTER(ctypes.c_int)),
                ('quant_number', ctypes.POINTER(ctypes.c_int)),
                ('data', ctypes.c_void_p),
                ('dev_mem', ctypes.c_void_p),
                ('mapped_mem', ctypes.c_void_p)]


class custom_kernel_ops(ctypes.Structure):
    pass


function_type_infer_shape = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(custom_kernel_ops),
                                             ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                             ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), ctypes.c_int,
                                             ctypes.c_int)
function_type_inplace_info = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(custom_kernel_ops), ctypes.c_int)

function_type_bind = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(custom_kernel_ops),
                                      ctypes.POINTER(ctypes.POINTER(custom_kernel_tensor)), ctypes.c_int,
                                      ctypes.POINTER(ctypes.POINTER(custom_kernel_tensor)), ctypes.c_int)

function_type_prerun = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(custom_kernel_ops),
                                        ctypes.POINTER(ctypes.POINTER(custom_kernel_tensor)),
                                        ctypes.c_int, ctypes.POINTER(ctypes.POINTER(custom_kernel_tensor)),
                                        ctypes.c_int, ctypes.c_int)

function_type_reshape = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(custom_kernel_ops),
                                         ctypes.POINTER(ctypes.POINTER(custom_kernel_tensor)),
                                         ctypes.c_int, ctypes.POINTER(ctypes.POINTER(custom_kernel_tensor)),
                                         ctypes.c_int)

function_type_run = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(custom_kernel_ops),
                                     ctypes.POINTER(ctypes.POINTER(custom_kernel_tensor)),
                                     ctypes.c_int, ctypes.POINTER(ctypes.POINTER(custom_kernel_tensor)), ctypes.c_int)

function_type_postrun = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(custom_kernel_ops),
                                         ctypes.POINTER(ctypes.POINTER(custom_kernel_tensor)),
                                         ctypes.c_int, ctypes.POINTER(ctypes.POINTER(custom_kernel_tensor)),
                                         ctypes.c_int)

function_type_release = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(custom_kernel_ops))

custom_kernel_ops._fields_ = [('kernel_name', ctypes.c_char_p),
                              ('op', ctypes.c_char_p),
                              ('force', ctypes.c_int),
                              ('kernel_param', ctypes.c_void_p),
                              ('kernel_param_size', ctypes.c_int),
                              ('infer_shape', function_type_infer_shape),
                              ('inplace_info', function_type_inplace_info),
                              ('bind', function_type_bind),
                              ('prerun', function_type_prerun),
                              ('reshape', function_type_reshape),
                              ('run', function_type_run),
                              ('postrun', function_type_postrun),
                              ('release', function_type_release)]


class cpu_cluster(ctypes.Structure):
    _fields_ = [('cpu_number', ctypes.c_int), ('max_freq', ctypes.c_int),
                ('cpu_model', ctypes.c_int), ('cpu_arch', ctypes.c_int),
                ('l1_size', ctypes.c_int), ('l2_size', ctypes.c_int),
                ('hw_cpu_id', ctypes.c_int * 8)]


class cpu_info(ctypes.Structure):
    _fields_ = [('cpu_name', ctypes.c_char_p), ('board_name', ctypes.c_char_p),
                ('cluster_number', ctypes.c_int), ('l3_size', ctypes.c_int),
                ('cpu_cluster', ctypes.POINTER(cpu_cluster)), ('online_cpu_number', ctypes.c_int),
                ('online_cpu_list', ctypes.c_int * 4)]
