#include "tengine_errno.hpp"

static __thread int tengine_errno;

void set_tengine_errno(int err_num)
{
    tengine_errno = err_num;
}

int get_tengine_errno(void)
{
    return tengine_errno;
}
