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
 * Copyright (c) 2020, Open AI Lab
 * Author: jjzeng@openailab.com
 */

#ifndef __TENGINE_OP_HPP__
#define __TENGINE_OP_HPP__

#include <list>
#include <memory>

namespace tengine
{
    class Tensor;
    namespace nn
    {
        class TengineOp
        {
            public:
                virtual ~TengineOp() {}
                virtual bool run() = 0;
                virtual bool valid()const = 0;
                virtual class Tensor* get_output_tensor()const = 0;
        };

        typedef std::shared_ptr<TengineOp> TTengineOpPtr;
        typedef std::list<TengineOp*> TTengineOpList;

        class TengineOpGraph : public TengineOp
        {
            public:
                TengineOpGraph()
                    {}
                virtual bool add_op(TengineOp* op) = 0;
            protected:
                TTengineOpList _op;
        };

    }
}

#endif
