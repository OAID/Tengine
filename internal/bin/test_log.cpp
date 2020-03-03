#include <unistd.h>
#include <string.h>

#include <thread>
#include <fstream>
#include <mutex>

#include "tengine_c_api.h"
#include "logger.hpp"

using namespace TEngine;

FILE* os_fp = nullptr;

int total_size = 0;
int total_count = 0;

void output_file(const char* s)
{
    int n = strlen(s);

    for(int i = 0; i < n; i++)
    {
        int c = s[i];

        if(c != '1' && c != 'a' && c != 'A' && c != '\n' && c != 'M' && c != 'Y')
        {
            printf("weird char [0x%x]\n", c);
        }
    }

    std::string buf("MY:");
    buf = buf + s;

    total_count++;
    total_size += buf.size();

    fwrite(buf.c_str(), buf.size(), 1, os_fp);
}

void output_stdout(const char* s)
{
    std::string buf("MY:");
    buf = buf + s;

    std::cout << buf;
}

void thread0_func(void)
{
    int i = 0;

    while(i++ < 1000)
    {
        LOG_INFO() << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1";
        LOG_INFO() << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1";
        LOG_INFO() << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1";
        LOG_INFO() << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1";
        LOG_INFO() << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1";
        LOG_INFO() << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "1"
                   << "\n";
    }
}

void thread1_func(void)
{
    int i = 0;

    while(i++ < 1000)
    {
        LOG_INFO() << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a";
        LOG_INFO() << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a";
        LOG_INFO() << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a";
        LOG_INFO() << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a";
        LOG_INFO() << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a";
        LOG_INFO() << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "a"
                   << "\n";
    }
}

void thread2_func(void)
{
    int i = 0;

    while(i++ < 1000)
    {
        LOG_INFO() << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A";
        LOG_INFO() << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A";
        LOG_INFO() << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A";
        LOG_INFO() << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A";
        LOG_INFO() << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A";
        LOG_INFO() << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "A"
                   << "\n";
    }
}

int main(int argc, char* argv[])
{
    if(argv[1])
    {
        os_fp = fopen(argv[1], "w");

        set_log_output(output_file);
    }
    else
    {
        set_log_output(output_stdout);
    }

    std::thread* tr0 = new std::thread(thread0_func);
    std::thread* tr1 = new std::thread(thread1_func);
    std::thread* tr2 = new std::thread(thread2_func);

    tr0->join();
    tr1->join();
    tr2->join();

    delete tr0;
    delete tr1;
    delete tr2;

    if(os_fp)
        fclose(os_fp);

    std::cout << "total_count: " << total_count << "\n";
    std::cout << "total_size: " << total_size << "\n";
    return 0;
}
