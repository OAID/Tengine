#!/usr/bin/env python

# coding: utf-8
from setuptools import setup, Extension

# from distutils.core import setup, Extension

files = ["__init__", "base", "context", "device", "graph", "libinfo", "node", "tengine", "tensor"]

setup(name="pytengine",
      version="0.9.1",
      description="Tengine is a software development kit for front-end intelligent devices developed by OPENAILAB",
      author="OpenAILab",
      author_email="OpenAILab@126.com",
      url="https://github.com/OAID/Tengine",
      packages=["tengine"],
      install_requires=['numpy>=1.4.0']
      )
