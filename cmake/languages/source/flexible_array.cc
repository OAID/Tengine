#include <stddef.h>

typedef struct foo
{
    int  length;
    char contents[];
} foo_t;

int main()
{
    const size_t size_of_array = sizeof(foo_t);
    const size_t size_of_integer = sizeof(int);

    return size_of_array == size_of_integer;
}
