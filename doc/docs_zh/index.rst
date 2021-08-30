.. Tengine documentation master file, created by
   sphinx-quickstart on Wed Mar 10 19:23:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tengine's documentation!
===================================


*请在页面左下角选择特定版本的文档。*

.. toctree::
  :maxdepth: 1
  :caption: 简介
  :name: sec-introduction

  introduction/tech_highlights
  introduction/architecture
  introduction/support_hardware
  introduction/support_operation_list
  introduction/support_model_list

.. toctree::
  :maxdepth: 1
  :caption: Benchmark
  :name: sec-benchmark
  
  benchmark/benchmark_tools

.. toctree::
  :maxdepth: 1
  :caption: 快速开始
  :name: sec-quick-start

  quick_start/tutorial
  quick_start/c_demo

.. toctree::
  :maxdepth: 1
  :caption: 使用工具
  :name: sec-user-guides

  user_guides/convert_tool
  user_guides/quant_tool_int8
  user_guides/quant_tool_uint8
  user_guides/visual_tool
  user_guides/debug

.. toctree::
  :maxdepth: 1
  :caption: 部署示例
  :name: sec-demo_guides

  demo_guides/linux_demo
  demo_guides/android_demo
  demo_guides/opencl
  demo_guides/timvx
  demo_guides/acl
  demo_guides/tensort
  demo_guides/cuda

.. toctree::
  :maxdepth: 1
  :caption: 源码编译
  :name: sec-source-compile

  source_compile/compile_env
  source_compile/compile_options
  source_compile/compile_linux
  source_compile/compile_vs
  source_compile/compile_android  
  source_compile/compile_acl
  source_compile/compile_cuda
  source_compile/compile_opencl  
  source_compile/compile_tensort
  source_compile/compile_timvx
  source_compile/compile_ohos
  source_compile/compile_visual_studio
  source_compile/compile_vulkan
  source_compile/demo_videocapture
  source_compile/deploy_SuperEdge
  source_compile/dla_opendla

.. toctree::
  :maxdepth: 1
  :caption: API文档

  api_reference/c_api_doc

.. toctree::
  :maxdepth: 1
  :caption: 开发者贡献

  develop_guides/architecture-intro
  develop_guides/add_hardware
  develop_guides/add_operator_opencl

.. toctree::
  :maxdepth: 1
  :caption: Roadmap
  :name: sec-roadmap

  introduction/roadmap
