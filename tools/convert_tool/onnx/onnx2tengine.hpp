#ifndef __ONNX2TENGINE_HPP__
#define __ONNX2TENGINE_HPP__

#include <string>
#include <iostream>
#include <fstream>
#include "onnx.pb.h"

extern "C" 
{
    #include "tengine/c_api.h"
    #include "graph/graph.h"
    #include "graph/subgraph.h"
    #include "graph/node.h"
    #include "graph/tensor.h"
    #include "executer/executer.h"
    #include "module/module.h"
    #include "utility/log.h"
    #include "utility/sys_port.h"
}


graph_t onnx2tengine(std::string model_file);


#endif