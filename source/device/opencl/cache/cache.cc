#pragma once

#include "cache.hpp"

void cl_cache::de_serializer(const std::string& cache_path)
{
    struct stat stat;

    int fd = open(cache_path.c_str(), O_RDONLY);

    if (fd < 0)
    {
        TLOG_ERR("cannot open file %s\n", cache_path.c_str());
        return;
    }

    fstat(fd, &stat);

    int file_len = stat.st_size;
    void* mem_base = (void*)sys_malloc(file_len);
    int ret = read(fd, mem_base, file_len);
    char* read_current = (char*)mem_base;

    uint16_t version = read<uint16_t>(&read_current);
    TLOG_ERR("current cache version is: %d \n", version);

    int auto_tune_size = read<int>(&read_current);
    if (auto_tune_size > 0)
    {
        std::vector<char> temp_key;
        auto_tune_vector.resize(auto_tune_size);
        for (int i = 0; i < auto_tune_size; ++i)
        {
            auto_tune temp_auto_tune{};
            int key_size = read<int>(&read_current);
            temp_key.resize(key_size);
            memcpy(temp_key.data(), read_current, key_size);
            std::string key(temp_key.begin(), temp_key.end());
            read_current += key_size;
            temp_auto_tune.key = key;
            temp_auto_tune.global_size[0] = read<int>(&read_current);
            temp_auto_tune.global_size[1] = read<int>(&read_current);
            temp_auto_tune.global_size[2] = read<int>(&read_current);
            temp_auto_tune.local_size[0] = read<int>(&read_current);
            temp_auto_tune.local_size[1] = read<int>(&read_current);
            temp_auto_tune.local_size[2] = read<int>(&read_current);
            auto_tune_vector[i] = temp_auto_tune;

            TLOG_ERR("decode cache: %s %d,%d,%d  %d,%d,%d \n",
                     key.c_str(),
                     temp_auto_tune.global_size[0],
                     temp_auto_tune.global_size[1],
                     temp_auto_tune.global_size[2],
                     temp_auto_tune.local_size[0],
                     temp_auto_tune.local_size[1],
                     temp_auto_tune.local_size[2]);
        }
    }

    close(fd);
}

void cl_cache::serializer(const std::string& cache_path)
{
    int fd = open(cache_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd == -1)
    {
        TLOG_ERR("Could not open %s\n", cache_path.c_str());
        return;
    }

    auto base = (char*)sys_malloc(get_auto_tune_size());
    auto out_ptr = base;
    write<uint16_t>(&out_ptr, CACHE_VERSION);
    write<int>(&out_ptr, auto_tune_vector.size());

    for (int i = 0; i < auto_tune_vector.size(); ++i)
    {
        write<int>(&out_ptr, auto_tune_vector[i].key.size());
        memcpy(out_ptr, auto_tune_vector[i].key.c_str(), auto_tune_vector[i].key.size());
        out_ptr += auto_tune_vector[i].key.size();
        write<int>(&out_ptr, auto_tune_vector[i].global_size[0]);
        write<int>(&out_ptr, auto_tune_vector[i].global_size[1]);
        write<int>(&out_ptr, auto_tune_vector[i].global_size[2]);
        write<int>(&out_ptr, auto_tune_vector[i].local_size[0]);
        write<int>(&out_ptr, auto_tune_vector[i].local_size[1]);
        write<int>(&out_ptr, auto_tune_vector[i].local_size[2]);
    }
    write(fd, base, get_auto_tune_size());
    close(fd);
}

int cl_cache::get_auto_tune_size()
{
    int size = 2;
    for (int i = 0; i < auto_tune_vector.size(); ++i)
    {
        size += 4 + 4 * 3 + 4 * 3;
        size += auto_tune_vector[i].key.size();
    }
    return size;
}
void cl_cache::test()
{
    int size = 10;
    for (int i = 0; i < size; ++i)
    {
        struct auto_tune temp
        {
        };
        temp.key = "ohoho" + std::to_string(i);
        temp.global_size[0] = i;
        temp.local_size[0] = i;
        auto_tune_vector.push_back(temp);
    }

    serializer("./cl.cache");

    de_serializer("./cl.cache");
}

int cl_cache::get_cache_tune(const std::string& key, auto_tune* tune)
{
    auto res = std::find_if(auto_tune_vector.begin(), auto_tune_vector.end(), [key](const auto_tune& left) {
        return left.key == key;
    });
    if (res != auto_tune_vector.end())
    {
        auto temp_auto_tune = *res;
        TLOG_ERR("find cache: %s %d,%d,%d  %d,%d,%d \n",
                 key.c_str(),
                 temp_auto_tune.global_size[0],
                 temp_auto_tune.global_size[1],
                 temp_auto_tune.global_size[2],
                 temp_auto_tune.local_size[0],
                 temp_auto_tune.local_size[1],
                 temp_auto_tune.local_size[2]);
        *tune = *res;
        return 0;
    }
    return -1;
}

void cl_cache::set_auto_tune(const auto_tune& tune)
{
    TLOG_ERR("add cache: %s %d,%d,%d  %d,%d,%d \n",
             tune.key.c_str(),
             tune.global_size[0],
             tune.global_size[1],
             tune.global_size[2],
             tune.local_size[0],
             tune.local_size[1],
             tune.local_size[2]);
    auto_tune_vector.push_back(tune);
    TLOG_ERR("cache size: %d \n", auto_tune_vector.size());
}
