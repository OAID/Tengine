/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: lswang@openailab.com
 */

#ifndef __EXEC_GRAPH_H__
#define __EXEC_GRAPH_H__

struct ir_graph;
struct subgraph;
struct vector;


int find_children_nodes(const struct subgraph* sub_graph, const uint16_t* node_id, struct vector* children_nodes);
int find_all_children_nodes(const struct subgraph* sub_graph, const uint16_t* node_id, struct vector* children_nodes);


int split_graph(struct ir_graph* ir_graph);

int optimize_graph(struct ir_graph* ir_graph, int precision);

#endif
