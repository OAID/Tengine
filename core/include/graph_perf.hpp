#ifndef __GRAPH_PERF_HPP__
#define __GRAPH_PERF_HPP__

#include <mutex>

#include "tengine_c_api.h"

namespace TEngine {

#define ATTR_GRAPH_PERF_STAT "GraphPerfStat"

struct GraphPerfMsg
{
    int action;
    struct perf_info** buf;
    int buf_size;
    int ret_number;
};

}    // namespace TEngine

#endif
