#ifndef __GRAPH_OPT_HPP__
#define __GRAPH_OPT_HPP__

#include <vector>
#include <map>
#include "stdio.h"
#include "string.h"
#include <string>
#include "math.h"
extern "C" {
#include "api/c_api.h"
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
#include "clip_param.h"
}

int graph_opt(graph_t graph);

/*!
 * @brief remove a node below specified node.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  pre_node_id: specific node index.
 * @param [in]  del_node_id: to be removed node index.
 *
 * @return  statue value, 0 success, other value failure.
 */
int delete_node(ir_graph_t* graph, int16_t pre_node_id, int16_t del_node_id);

/*!
 * @brief add a node above specified node.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  down_node_id: specific node index.
 * @param [in]  add_node_type: to be added node op type.
 * @param [in]  name: to be added node name.
 *
 * @return  added node index.
 */
int add_node_above(ir_graph_t* graph, int16_t down_node_id, int add_node_type, const char* name);

/*!
 * @brief add a const node above specified node.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  down_node_id: specific node index.
 * @param [in]  add_node_type: to be added node op type.
 * @param [in]  name: to be added node name.
 *
 * @return  added node index.
 */
int add_const_node_above(ir_graph_t* graph, int16_t down_node_id, const char* name);

/*!
 * @brief add a node below specified node.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  down_node_id: specific node index.
 * @param [in]  name: to be added node name.
 *
 * @return  added const node index.
 */
int add_node_below(ir_graph_t* graph, int16_t up_node_id, int add_node_type, const char* name);

#endif
