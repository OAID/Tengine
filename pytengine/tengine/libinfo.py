
# coding: utf-8
"""Information about Tengine."""
from __future__ import absolute_import
import os
import platform
import logging


def find_lib_path():
    """Find Tengine dynamic library files.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries.
    """
    dll_path = []
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path.append(curr_path)
    #dll_path.append(os.path.join(curr_path,"../build"))
    dll_path.append(os.path.join(curr_path,"lib"))
    for path in os.environ['LD_LIBRARY_PATH'].split(':'):
        dll_path.append(path)
        dll_path.append(os.path.join(path,"build"))
        dll_path.append(os.path.join(path, "install/lib"))
    path = [os.path.join(p,"libtengine.so") for p in dll_path]
    lib_path = [p for p in path if os.path.exists(p) and os.path.isfile(p)]
    return lib_path

def find_include_path():
    """Find Tengine included header files.

    Returns
    -------
    incl_path : string
        Path to the header files.
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # include path in pip package
    pip_incl_path = os.path.join(curr_path, 'include/')
    if os.path.isdir(pip_incl_path):
        return pip_incl_path
    else:
        # include path if build from source
        src_incl_path = os.path.join(curr_path, '../../include/')
        if os.path.isdir(src_incl_path):
            return src_incl_path
        else:
            raise RuntimeError('Cannot find the Tengine include path in either ' + pip_incl_path +
                               ' or ' + src_incl_path + '\n')


# current version
__version__ = "0.1.0"
