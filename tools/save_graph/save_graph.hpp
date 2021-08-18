#pragma once

#include <stdlib.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <fcntl.h>
#include <functional>

extern "C" {
#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "utility/log.h"
#include "operator/op.h"
#include "serializer/tmfile/tm2_format.h"
}

#include "tm2_op_save.hpp"

bool save_graph(graph_t graph, const char* fname);
