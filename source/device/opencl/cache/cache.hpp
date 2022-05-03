#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/file.h>
#include <unistd.h>

#include "ocl_cpp_helper.hpp"
#include "utility/sys_port.h"

struct auto_tune
{
    std::string key;
    std::vector<int> global_size = {0, 0, 0};
    std::vector<int> local_size = {0, 0, 0};
};

#define CACHE_VERSION 1

class cl_cache
{
public:
    cl_cache()
    {
        auto_tune_vector.clear();
    };
    ~cl_cache() = default;

    void test();

    void de_serializer(const std::string& path);
    void serializer(const std::string& path);

    int get_cache_tune(const std::string& key, auto_tune* tune);
    void set_auto_tune(const auto_tune& tune);

private:
    std::vector<auto_tune> auto_tune_vector;
    int get_auto_tune_size();
};

template<typename Tp>
static inline Tp read(char** current)
{
    auto tpr = (Tp*)*current;
    *current += sizeof(Tp);
    return *tpr;
}

template<typename Tp>
static inline void write(char** current, Tp value)
{
    auto tpr = (Tp*)*current;
    tpr[0] = value;
    *current += sizeof(Tp);
}