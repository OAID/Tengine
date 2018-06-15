FILE(GLOB_RECURSE core_src "core/lib/*.cpp")

include_directories(core/include)
include_directories(include)

FOREACH(src ${core_src})
list(APPEND TENGINE_LIB_SRCS ${src})
ENDFOREACH()

