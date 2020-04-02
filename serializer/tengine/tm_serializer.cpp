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
 * Copyright (c) 2019, Open AI Lab
 * Author: jingyou@openailab.com
 */
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "operator_manager.hpp"
#include "static_graph.hpp"
#include "graph.hpp"

#include "tm_serializer.hpp"

#define TM_FILE_MAX_SIZE 1 << 30 /* 1G */

namespace TEngine {

// template SpecificFactory<TmSerializer> SpecificFactory<TmSerializer>::instance;
template class SpecificFactory<TmSerializer>;

extern bool register_tm1_serializer();
extern bool register_tm2_serializer();

bool TmSerializer::SaveModel(const std::vector<std::string>& file_list, Graph* graph)
{
    /* Check the file number */
    if(file_list.size() != GetFileNum())
        return false;

    /* Open the tengine model file */
    int fd = open(file_list[0].c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if(fd == -1)
    {
        LOG_ERROR() << "Could not open " << file_list[0] << "\n";
        return false;
    }

    std::vector<void*> addr_list;
    std::vector<int> size_list;

    if(!SaveModel(addr_list, size_list, graph))
    {
        close(fd);
        return false;
    }

    void* buf = addr_list[0];
    int size = size_list[0];
    int ret = write(fd, buf, size);

    close(fd);
    free(buf);

    if(ret != size)
        return false;
    else
        return true;
}

bool TmSerializer::SaveModel(std::vector<void*>& addr_list, std::vector<int>& size_list, Graph* graph)
{
    uint32_t tm_model_size = 0;

    uint32_t malloc_size = TM_FILE_MAX_SIZE;
    const char* env = std::getenv("TM_FILE_MAX_SIZE");
    if(env)
        malloc_size = std::atoi(env);

    void* start_ptr = ( void* )malloc(malloc_size);
    if(start_ptr == nullptr)
    {
        LOG_ERROR() << "Malloc memory failed: " << malloc_size << ".\n";
        return false;
    }

    TmSerializerPtr tm_serializer;
    TmSerializerManager::SafeGet("tm_v2", tm_serializer);

    bool ret = tm_serializer->SaveModelIntoMem(start_ptr, graph, &tm_model_size);

    addr_list.push_back(start_ptr);
    size_list.push_back(tm_model_size);

    return ret;
}

bool TmSerializer::LoadBinaryFile(const char* tm_fname, int& fd, void*& buf, int& size)
{
    fd = open(tm_fname, O_RDONLY);
    if(fd == -1)
    {
        LOG_ERROR() << "Could not open \'" << tm_fname << "\'\n";
        return false;
    }

    struct stat sb;
    fstat(fd, &sb);
    size = sb.st_size;

    buf = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if(buf == MAP_FAILED)
    {
        LOG_ERROR() << "Mmap of \'" << tm_fname << "\' failed\n";
        return false;
    }

    return true;
}

bool TmSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    int fd;
    void* mmap_buf;
    int mmap_size;

    if(file_list.size() != GetFileNum())
        return false;

    if(!LoadBinaryFile(file_list[0].c_str(), fd, mmap_buf, mmap_size))
        return false;

    SetGraphSource(graph, file_list[0]);
    SetGraphSourceFormat(graph, "tengine");
    SetGraphConstTensorFile(graph, file_list[0]);

    const uint16_t* ver_main = reinterpret_cast<const uint16_t*>(mmap_buf);
    TmSerializerPtr tm_serializer;
    if(*ver_main < 2)
    {
        LOG_WARN()
            << "The input tengine model file is in old format, please regenerate it by using tengine convert tool.\n";
        TmSerializerManager::SafeGet("tm_v1", tm_serializer);
    }
    else
        TmSerializerManager::SafeGet("tm_v2", tm_serializer);

    bool ret = tm_serializer->LoadModelFromMem(mmap_buf, graph);

    munmap(const_cast<void*>(mmap_buf), mmap_size);
    close(fd);
    return ret;
}

bool TmSerializer::LoadModel(const std::vector<const void*>& addr_list, const std::vector<int>& size_list,
                             StaticGraph* graph, bool transfer_mem)
{
    if(addr_list.size() != GetFileNum())
        return false;

    void* mmap_buf = ( void* )addr_list[0];

    SetGraphSource(graph, "in_mem");
    SetGraphSourceFormat(graph, "tengine");

    const uint16_t* ver_main = reinterpret_cast<const uint16_t*>(mmap_buf);
    TmSerializerPtr tm_serializer;
    if(*ver_main < 2)
    {
        LOG_WARN()
            << "The input tengine model file is in old format, please regenerate it by using tengine convert tool.\n";
        TmSerializerManager::SafeGet("tm_v1", tm_serializer);
    }
    else
        TmSerializerManager::SafeGet("tm_v2", tm_serializer);

    bool ret = tm_serializer->LoadModelFromMem(mmap_buf, graph);

    if(ret && transfer_mem)
        graph->mem_src.push_back(mmap_buf);

    return ret;
}

bool TmSerializerInit(void)
{
    auto factory = SerializerFactory::GetFactory();

    factory->RegisterInterface<TmSerializer>("tengine");
    auto tm_serializer = factory->Create("tengine");

    SerializerManager::SafeAdd("tengine", SerializerPtr(tm_serializer));

    bool ret1 = register_tm1_serializer();
    bool ret2 = register_tm2_serializer();

    return (ret1 && ret2);
}

}    // namespace TEngine
