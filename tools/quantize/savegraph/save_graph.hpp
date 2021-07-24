#pragma once

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <unordered_map>
#include <fcntl.h>
#include <functional>

extern "C" {
#include "tengine/c_api.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "utility/log.h"
#include "operator/op.h"
#include "tm2_format.h"
}

#include "tm2_op_save.hpp"

bool save_graph(graph_t graph, const char* fname);
