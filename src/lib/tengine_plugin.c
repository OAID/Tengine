#include <stdio.h>
#include <string.h>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "compiler.h"
#include "sys_port.h"
#include "tengine_c_api.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "vector.h"

typedef const char* const_char_t;

#ifdef _MSC_VER
typedef int(*fun_ptr)(void);
typedef HINSTANCE so_handle_t;
#else
typedef void* so_handle_t;
#endif

struct plugin_header
{
    char* name;
    char* fname;
    so_handle_t handle;
};

static struct vector* plugin_list = NULL;

static int exec_so_func(so_handle_t handle, const char* func_name)
{
#ifdef _MSC_VER
	void* func = (fun_ptr)GetProcAddress(handle, func_name);
#else
	void* func = dlsym(handle, func_name);
#endif

    if (func == NULL)
    {
#ifdef _MSC_VER
		TLOG_ERR("find func: %s failed, error code %d\n", func_name, GetLastError());
#else
		TLOG_ERR("find func: %s failed, reason %s\n", func_name, dlerror());
#endif

        return -1;
    }

    int (*call_func)(void) = func;

    if (call_func() < 0)
    {
        TLOG_ERR("exec so func: %s failed\n", func_name);
        return -1;
    }
    TLOG_INFO("function:%s executed\n", func_name);

    return 0;
}

int load_tengine_plugin(const char* plugin_name, const char* fname, const char* init_func_name)
{
    struct plugin_header header;

    /* TODO: MT safe */

    if (plugin_list == NULL)
    {
        plugin_list = create_vector(sizeof(struct plugin_header), NULL);

        if (plugin_list == NULL)
        {
            set_tengine_errno(ENOMEM);
            return -1;
        }
    }

    /* check if name duplicated */
    int list_num = get_vector_num(plugin_list);

    for (int i = 0; i < list_num; i++)
    {
        struct plugin_header* h = get_vector_data(plugin_list, i);

        if (!strcmp(h->name, plugin_name))
        {
            TLOG_ERR("duplicated plugin name: %s\n", plugin_name);
            set_tengine_errno(EEXIST);
            return -1;
        }
    }

    /* load the so */
#ifdef _MSC_VER
	header.handle = LoadLibraryA(fname);
#else
	header.handle = dlopen(fname, RTLD_LAZY);
#endif

    if (header.handle == NULL)
    {
#ifdef _MSC_VER
		TLOG_ERR("load plugin failed: error code %d\n", GetLastError());
#else
		TLOG_ERR("load plugin failed: %s\n", dlerror());
#endif
        set_tengine_errno(EINVAL);
        return -1;
    }

    /* execute the init function */
    if (init_func_name && exec_so_func(header.handle, init_func_name) < 0)
    {
        set_tengine_errno(EINVAL);

#ifdef _MSC_VER
		FreeLibrary(header.handle);
#else
		dlclose(header.handle);
#endif

        return -1;
    }

    header.name = strdup(plugin_name);
    header.fname = strdup(fname);

    push_vector_data(plugin_list, &header);

    return 0;
}

int unload_tengine_plugin(const char* plugin_name, const char* rel_func_name)
{
    if (plugin_list == NULL)
        return -1;

    int list_num = get_vector_num(plugin_list);
    struct plugin_header* target = NULL;

    for (int i = 0; i < list_num; i++)
    {
        struct plugin_header* h = get_vector_data(plugin_list, i);

        if (!strcmp(h->name, plugin_name))
        {
            target = h;
            break;
        }
    }

    if (target == NULL)
    {
        set_tengine_errno(ENOENT);
        return -1;
    }

    if (rel_func_name)
        exec_so_func(target->handle, rel_func_name);

#ifdef _MSC_VER
	FreeLibrary(target->handle);
#else
	dlclose(target->handle);
#endif

    remove_vector_data(plugin_list, target);

    if (get_vector_num(plugin_list) == 0)
    {
        release_vector(plugin_list);
    }

    return 0;
}

int get_tengine_plugin_number(void)
{
    int plugin_num = 0;

    if (plugin_list)
        plugin_num = get_vector_num(plugin_list);

    return plugin_num;
}

const_char_t get_tengine_plugin_name(int idx)
{
    int plugin_num = get_tengine_plugin_number();

    if (idx >= plugin_num)
        return NULL;

    struct plugin_header* h = get_vector_data(plugin_list, idx);

    return h->name;
}
