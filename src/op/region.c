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
 * Author: qtang@openailab.com
 */

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "region_param.h"

DEFINE_PARM_PARSE_ENTRY(region_param, num_classes, side, num_box, coords, confidence_threshold, nms_threshold,
                        biases_num, biases);

static int init_op(struct ir_op* op)
{
    struct region_param* region_param = ( struct region_param* )sys_malloc(sizeof(struct region_param));

    if (region_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    region_param->num_classes = 1;

    op->param_mem = region_param;
    op->param_size = sizeof(struct region_param);
    op->same_shape = 1;
    op->infer_shape = NULL;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_region_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_REGION, OP_REGION_NAME, &m);
}

static int unregister_region_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(region_param));
    return unregister_op(OP_REGION, 1);
}

AUTO_REGISTER_OP(register_region_op);
AUTO_UNREGISTER_OP(unregister_region_op);
