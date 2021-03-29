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
 * Author: bsun@openailab.com
 */

#include "nnie_opmaps.h"
#include "tengine_nnie_plugin.h"
using namespace std;
namespace Tengine {
    NnieOpMaps::~NnieOpMaps(){
        custom_ops.clear();
        nnieOpMaps=nullptr;
    }
    NnieOpMaps* NnieOpMaps::getInstance(){
        return nnieOpMaps;
    }
    void NnieOpMaps::addCustomOp(struct custom_op* op){
        custom_ops.insert(std::pair<std::string,custom_op*>(string(op->name),op));
    }
    void NnieOpMaps::removeCustionOp(struct custom_op* op){
        custom_ops.erase(string(op->name));
    }
    bool NnieOpMaps::exist(struct custom_op* op){
        if(custom_ops.find(op->name)==custom_ops.end())
            return false;
        else
            return true;

    }
    struct custom_op* NnieOpMaps::getCustomOpByName(std::string name){
        std::unordered_map<std::string,custom_op*>::iterator it = custom_ops.find(name);
        if(it==custom_ops.end())
            return nullptr;
        else
            return it->second;
    }

    NnieOpMaps::NnieOpMaps(){

    }

    NnieOpMaps* NnieOpMaps::nnieOpMaps = new NnieOpMaps();

}// namespace Tengine
