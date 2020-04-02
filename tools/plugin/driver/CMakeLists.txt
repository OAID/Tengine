include_directories(${CONFIG_TENGINE_ROOT}/include)
include_directories(${CONFIG_TENGINE_ROOT}/core/include)
include_directories(${CONFIG_TENGINE_ROOT}/operator/include)
include_directories(${CONFIG_TENGINE_ROOT}/executor/include)

if(TENGINE_IN_BIND)
	set(TENGINE_LIBRARY tengine)
else()
	set(TENGINE_LIBRARY ${CONFIG_TENGINE_ROOT}/build/libtengine.so)
endif()

#the overall version for all drivers
set(DRIVER_VERSION 1)

#the so version for versoin switch
set(DRIVER_SO_VERSION 1)


if(CONFIG_ACL_OPENCL)
    add_subdirectory(acl_opencl)
endif()


