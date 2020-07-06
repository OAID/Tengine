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
#include <stdlib.h>

#include "sys_port.h"
#include "vector.h"
#include "parameter.h"

#include "tengine_ir.h"
#include "tengine_op.h"
#include "tengine_errno.h"
#include "tengine_utils.h"

static struct vector* op_list;

int init_op_registry(void)
{
    op_list = create_vector(sizeof(struct op_method), NULL);

    if (op_list == NULL)
        return -1;

    return 0;
}

void release_op_registry(void)
{
    release_vector(op_list);
}

static int register_op_registry(struct op_method* op_method)
{
    if (find_op_method(op_method->op_type, op_method->op_version))
    {
        set_tengine_errno(EEXIST);
        return -1;
    }

    return push_vector_data(op_list, op_method);
}

struct op_method* find_op_method(int op_type, int op_version)
{
    int num = get_vector_num(op_list);

    for (int i = 0; i < num; i++)
    {
        struct op_method* m = get_vector_data(op_list, i);

        if (m->op_type == op_type)
            return m;
    }

    return NULL;
}

int register_op(int op_type, const char* op_name, struct op_method* op_method)
{
    if (op_name && register_op_map(op_type, op_name) < 0)
        return -1;

    if (op_method)
    {
        op_method->op_type = op_type;
        return register_op_registry(op_method);
    }

    return 0;
}

int unregister_op(int op_type, int version)
{
    int num = get_vector_num(op_list);
    int op_count = 0;
    int target_idx = -1;

    for (int i = 0; i < num; i++)
    {
        struct op_method* m = get_vector_data(op_list, i);

        if (m->op_type == op_type)
        {
            op_count++;

            if (m->op_version == version)
                target_idx = i;
        }
    }

    if (target_idx < 0)
        return -1;

    remove_vector_by_idx(op_list, target_idx);

    op_count--;

    /* if last version of this op is removed, removing the op name mapping */
    if (op_count == 0)
        unregister_op_map(op_type);

    return 0;
}
