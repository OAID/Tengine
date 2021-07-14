#ifndef __GRAPH_OPT_HPP__
#define __GRAPH_OPT_HPP__

#include <vector>
#include <map>
#include "stdio.h"
#include "string.h"
#include <string>
#include "math.h"
extern "C" 
{
    #include "tengine/c_api.h"
    #include "graph/graph.h"
    #include "graph/node.h"
    #include "graph/tensor.h"
    #include "module/module.h"
    #include "utility/log.h"
    #include "utility/sys_port.h"
    
    #include "convolution_param.h"
    #include "relu_param.h"
    #include "eltwise_param.h"
    #include "batchnorm_param.h"
    #include "fc_param.h"
}

int graph_opt(graph_t graph);

#endif