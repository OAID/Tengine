#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <malloc.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

unsigned long get_cur_time(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (tv.tv_sec * 1000000 + tv.tv_usec);
}

void run_test(int reptition, int warm_up, void (*test_func)(void*), void* arg)
{
    if(warm_up)
        test_func(arg);

    for(int i = 0; i < reptition; i++)
    {
        test_func(arg);
    }
}

extern void test_mla_4s(long n);
extern void test_mla_16b(long n);
extern void test_fmla_4s(long n);
extern void test_mla_8h(long n);

#define TESTBED_WRAP(inst, a, b)                 \
    void testbed_wrap_##inst##_##a##b(void* arg) \
    {                                            \
        long n = ( long )arg;                    \
        test_##inst##_##a##b(n);                 \
    }

TESTBED_WRAP(mla, 4, s)
TESTBED_WRAP(mla, 8, h)
TESTBED_WRAP(mla, 16, b)
TESTBED_WRAP(fmla, 4, s)

int main(int argc, char* argv[])
{
    long n = 1024 * 16;

    unsigned long start;
    unsigned long end;

    float fops;

    start = get_cur_time();
    run_test(16, 0, testbed_wrap_mla_4s, ( void* )n);
    end = get_cur_time();

    fops = n * 16 * 1020 * 8;
    printf("MLA_4S: reptition %ld, used %lu us, calculate %.2f Mops\n", n, end - start,
           fops * (1000000.0 / (end - start)) / 1000000);

    start = get_cur_time();
    run_test(16, 0, testbed_wrap_mla_8h, ( void* )n);
    end = get_cur_time();

    fops = n * 16 * 1020 * 16;

    printf("MLA_8H: reptition %ld, used %lu us, calculate %.2f Mops\n", n, end - start,
           fops * (1000000.0 / (end - start)) / 1000000);

    start = get_cur_time();
    run_test(16, 0, testbed_wrap_mla_16b, ( void* )n);
    end = get_cur_time();

    fops = n * 16 * 1020 * 32;

    printf("MLA_16B: reptition %ld, used %lu us, calculate %.2f Mops\n", n, end - start,
           fops * (1000000.0 / (end - start)) / 1000000);

    start = get_cur_time();
    run_test(16, 0, testbed_wrap_fmla_4s, ( void* )n);
    end = get_cur_time();

    fops = n * 16 * 1020 * 8;

    printf("FMLA_4S: reptition %ld, used %lu us, calculated %.2f Mfops\n", n, end - start,
           fops * (1000000.0 / (end - start)) / 1000000);

    return 0;
}
