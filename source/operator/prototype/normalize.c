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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#include "normalize_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int init_op(struct op* op)
{
    normalize_param_t* normalize_param = ( normalize_param_t* )sys_malloc(sizeof(normalize_param_t));

    if (normalize_param == NULL)
    {
        return -1;
    }

    normalize_param->across_spatial = 0;
    normalize_param->channel_shared = 0;

    op->param_mem = normalize_param;
    op->param_size = sizeof(normalize_param_t);
    op->same_shape = 1;
    op->infer_shape = NULL;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_normalize_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_NORMALIZE, OP_NORMALIZE_NAME, &m);
}


int unregister_normalize_op()
{
    return unregister_op(OP_NORMALIZE, 1);
}
