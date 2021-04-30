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

#include "cast_param.h"

#include "graph/node.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int init_op(ir_op_t* op)
{
    struct cast_param* cast_param = (struct cast_param*)sys_malloc(sizeof(struct cast_param));
    if (cast_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    cast_param->type_from = 0;
    cast_param->type_to = 0;

    op->param_mem = cast_param;
    op->param_size = sizeof(struct cast_param);
    op->same_shape = 1;
    op->infer_shape = NULL;

    return 0;
}


static void release_op(ir_op_t* op)
{
    sys_free(op->param_mem);
}


int register_cast_op()
{
    ir_method_t m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_CAST, OP_CAST_NAME, &m);
}


int unregister_cast_op()
{
    return unregister_op(OP_CAST, 1);
}
