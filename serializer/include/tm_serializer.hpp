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
 * Copyright (c) 2018, Open AI Lab
 * Author: jingyou@openailab.com
 */
#ifndef __TM_SERIALIZER_HPP__
#define __TM_SERIALIZER_HPP__

#include "logger.hpp"
#include "serializer.hpp"
#include "static_graph_interface.hpp"
#include "tm_generate.h"

namespace TEngine {

class TmSerializer : public Serializer {
  using name_map_t = std::unordered_map<std::string, unsigned int>;

 public:
  TmSerializer() {
    name_ = "tm_loader";
    version_ = "0.1";
    format_name_ = "tengine";
  }

  virtual ~TmSerializer(){};

  unsigned int GetFileNum(void) override { return 1; }

  bool LoadModel(const std::vector<std::string> &file_list,
                 StaticGraph *graph) override;
  bool SaveModel(const std::vector<std::string> &file_list,
                 Graph *graph) override;

  bool LoadConstTensor(const std::string &fname,
                       StaticTensor *const_tensor) override {
    return false;
  }
  bool LoadConstTensor(int fd, StaticTensor *const_tensor) override {
    return false;
  }

 protected:
  bool LoadBinaryFile(const char *tm_fname);
  bool LoadNode(StaticGraph *graph, StaticNode *node, const TM_Node *tm_node);
  bool LoadTensor(StaticGraph *graph, const TM_Tensor *tm_tensor,
                  const TM_Buffer *tm_buf);
  bool LoadGraph(StaticGraph *graph, const TM_Model *tm_model);

  tm_uoffset_t SaveTmSubgraph(void *const start_ptr, tm_uoffset_t *cur_pos,
                              Graph *graph);
  tm_uoffset_t SaveTmNode(void *const start_ptr, tm_uoffset_t *cur_pos,
                          Node *node, name_map_t &tensor_name_map);
  tm_uoffset_t SaveTmTensor(void *const start_ptr, tm_uoffset_t *cur_pos,
                            Tensor *tensor, unsigned int tensor_id,
                            unsigned int buffer_id);

 private:
  int mmap_fd_;
  void *mmap_buf_;
  size_t mmap_buf_size_;

  bool tm_no_data_;
  bool tm_with_string_;
};

}  // namespace TEngine

#endif
