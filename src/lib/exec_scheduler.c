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
 * Author: haitao@openailab.com
 */

#include <stdio.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "exec_scheduler.h"
#include "nn_device.h"

static int sched_prerun(struct exec_scheduler* scheduler, struct ir_graph* ir_graph, int num_thread, int cpu_affinity, int mode)
{
    int subgraph_num = get_vector_num(ir_graph->subgraph_list);

    for (int i = 0; i < subgraph_num; i++)
    {
        struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, i);
        struct nn_device* nn_dev = subgraph->nn_dev;

        if (nn_dev->prerun(nn_dev, subgraph, num_thread, cpu_affinity, mode) < 0)
        {
            subgraph->status = GRAPH_STAT_ERROR;
            TLOG_ERR("subgraph %d prerun failed\n", subgraph->idx);

            return -1;
        }

        subgraph->status = GRAPH_STAT_READY;
    }

    return 0;
}

static int sched_run(struct exec_scheduler* scheduler, struct ir_graph* ir_graph, int block)
{
    if (block == 0)
    {
        TLOG_DEBUG("sync scheduler does not support non block run\n");
        set_tengine_errno(ENOTSUP);
        return -1;
    }

    struct vector* wait_list = create_vector(sizeof(struct subgraph*), NULL);

    if (wait_list == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    int subgraph_num = get_vector_num(ir_graph->subgraph_list);

    /* insert all subgraphs into wait list */

    for (int i = 0; i < subgraph_num; i++)
    {
        struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, i);

        push_vector_data(wait_list, &subgraph);
    }

    int* ready_list = sys_malloc(sizeof(int) * subgraph_num);

    if (ready_list == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    while (1)
    {
        int ready_num = 0;
        int wait_num = get_vector_num(wait_list);

        if (wait_num == 0)
            break;

        for (int i = 0; i < wait_num; i++)
        {
            struct subgraph* subgraph = *( struct subgraph** )get_vector_data(wait_list, i);

            if (subgraph->input_ready_count == subgraph->input_wait_count)
                ready_list[ready_num++] = i;
        }

        if (ready_num == 0)
        {
            TLOG_ERR("no sugraph is ready, while still %d subgraph in wait_list\n", wait_num);
            set_tengine_errno(EFAULT);
            return -1;
        }

        for (int i = 0; i < ready_num; i++)
        {
            struct subgraph* subgraph = *( struct subgraph** )get_vector_data(wait_list, ready_list[i]);
            struct nn_device* nn_dev = subgraph->nn_dev;

            subgraph->status = GRAPH_STAT_RUNNING;

            if (nn_dev->run(nn_dev, subgraph) < 0)
            {
                TLOG_ERR("run subgraph %d error!\n", subgraph->idx);
                sys_free(ready_list);
                release_vector(wait_list);
                subgraph->status = GRAPH_STAT_ERROR;
                return -1;
            }

            for (int j = 0; j < get_vector_num(ir_graph->subgraph_list); j++)
            {
                struct subgraph* waiting_sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, j);
                for (int k = 0; k < waiting_sub_graph->input_num; k++)
                {
                    uint16_t waiting_input_idx = waiting_sub_graph->input_tensor_list[k];
                    for (int m = 0; m < subgraph->output_num; m++)
                    {
                        int16_t current_output_idx = subgraph->output_tensor_list[m];
                        if (current_output_idx == waiting_input_idx)
                        {
                            waiting_sub_graph->input_ready_count++;
                        }
                    }
                }
            }

            subgraph->status = GRAPH_STAT_READY;
        }

        /* remove executed subgraph from list,
           shoudl from higher idx to lower idx */
        for (int i = ready_num - 1; i >= 0; i--)
            remove_vector_by_idx(wait_list, ready_list[i]);

#ifndef _MSC_VER
#ifdef __CC_ARM
        __memory_changed();
#else
        /* GNU CC*/
        __asm__ __volatile__("" ::: "memory"); /* force to read vector->num from memory */
#endif
#endif
    }

    /* reset the ready count */

    for (int i = 0; i < subgraph_num; i++)
    {
        struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, i);
        subgraph->input_ready_count = 0;
    }

    sys_free(ready_list);
    release_vector(wait_list);

    return 0;
}

static int sched_wait(struct exec_scheduler* scheduler, struct ir_graph* ir_graph)
{
    set_tengine_errno(ENOTSUP);
    return -1;
}

static int sched_postrun(struct exec_scheduler* scheduler, struct ir_graph* ir_graph)
{
    int subgraph_num = get_vector_num(ir_graph->subgraph_list);
    int has_error = 0;

    for (int i = 0; i < subgraph_num; i++)
    {
        struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, i);
        struct nn_device* nn_dev = subgraph->nn_dev;

        subgraph->status = GRAPH_STAT_DONE;

        if (nn_dev->postrun(nn_dev, subgraph) < 0)
        {
            subgraph->status = GRAPH_STAT_ERROR;
            has_error = 1;
            TLOG_ERR("sched %d prerun failed\n", subgraph->idx);
        }
    }

    if (has_error)
        return -1;
    else
        return 0;
}

static struct exec_scheduler sync_scheduler = {
    .name = "sync",
    .prerun = sched_prerun,
    .run = sched_run,
    .wait = sched_wait,
    .postrun = sched_postrun,
    .release = NULL,
};

struct exec_scheduler* get_default_scheduler(void)
{
    return &sync_scheduler;
}
