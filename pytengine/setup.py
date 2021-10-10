#!/usr/bin/env python

# coding: utf-8
from setuptools import setup, Extension

# from distutils.core import setup, Extension

import os, shutil
import platform

source = os.getcwd()
father_path = os.path.abspath(os.path.dirname(source) + os.path.sep + ".")
dest = os.getcwd()

libtengine = ""
if platform.system() == "Windows":
    libtengine = "tengine-lite.dll"
else:
    libtengine = "libtengine-lite.so"

src = father_path + "/build/install/lib/" + libtengine
dst = dest + "/tengine/" + libtengine
shutil.copyfile(src, dst)

setup(
    name="pytengine",
    version="0.9.1",
    description="Tengine is a software development kit for front-end intelligent devices developed by OPENAILAB",
    author="OpenAILab",
    author_email="OpenAILab@126.com",
    url="https://github.com/OAID/Tengine",
    packages=["tengine"],
    package_data={"tengine": [libtengine]},
    install_requires=["numpy>=1.4.0"],
)
