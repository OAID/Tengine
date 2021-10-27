#pragma once

struct tensor;
struct subgraph;

#define TENGINE_DUMP_DIR   "TG_DEBUG_DUMP_DIR"
#define TENGINE_DUMP_LAYER "TG_DEBUG_DATA"

void dump_sub_graph_trt(struct subgraph* sub_graph);