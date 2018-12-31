#ifndef __NODE_DUMP_HPP__
#define __NODE_DUMP_HPP__

#include <mutex>

namespace TEngine {

#define ATTR_GRAPH_NODE_DUMP "GraphNodeDump"

struct NodeDumpMsg
{
    int action;
    const char* node_name;
    void** buf;
    int buf_size;
    int ret_number;
};

}    // namespace TEngine

#endif
